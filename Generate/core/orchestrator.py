"""Оркестратор генерации клипов (этап 8, часть 2; SPEC 3, 4, 8.2, 8.3, 9, 0.1, 0.2, 0.5).

Конвейер: missing.txt -> проверка Release -> промпты Gemini -> аренда карт Vast.ai ->
рендер на воркерах -> немедленная выдача в Release -> уничтожение карт.

Только стандартная библиотека + httpx. Все сообщения на русском (SPEC 0.5).
Секреты (токен воркера, ключи API) не попадают в логи, repr и тексты ошибок (SPEC 0.1).
Время и пауза внедряются (clock/sleep), поэтому модуль детерминирован в тестах.

Протокол воркера (HTTP, порт 8000, заголовок X-Worker-Token):
  POST /task          {num, prompt, num_frames, seed} -> {"id": ...}
  GET  /task/{id}     -> {"state"|"status": "queued|running|done|error|failed", "error": ...}
  GET  /file/{id}     -> байты mp4
  POST /ack/{id}      воркер удаляет клип со своего диска
"""
from __future__ import annotations

import json
import logging
import math
import os
import secrets
import shutil
import tempfile
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from . import gemini_orchestrator as go
    from . import gemini_prompts as gp
    from .budget import BudgetError, BudgetTracker
    from .missing import parse_missing
    from .naming import clip_filename
    from .release_adapter import (  # noqa: F401
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseInfo, ReleaseStore, ReleaseUploader,
    )
    from .vast_client import (  # noqa: F401
        WORKER_HTTP_PORT, GpuOffer, InstanceInfo, OfferFilter, VastClient, VastError,
    )
except ImportError:  # pragma: no cover
    import gemini_orchestrator as go  # type: ignore
    import gemini_prompts as gp  # type: ignore
    from budget import BudgetError, BudgetTracker  # type: ignore
    from missing import parse_missing  # type: ignore
    from naming import clip_filename  # type: ignore
    from release_adapter import (  # type: ignore  # noqa: F401
        LINKS_FILENAME, PROMPTS_FILENAME, ReleaseInfo, ReleaseStore, ReleaseUploader,
    )
    from vast_client import (  # type: ignore  # noqa: F401
        WORKER_HTTP_PORT, GpuOffer, InstanceInfo, OfferFilter, VastClient, VastError,
    )

log = logging.getLogger("Generate.orchestrator")

MAX_ATTEMPTS = 3          # попыток рендера одного клипа
MAX_NET_FAILS = 6         # подряд сбоев связи с воркером до снятия карты
EXIT_OK = 0
EXIT_PARTIAL = 3

_DONE_STATUSES = frozenset({"done", "completed", "ready"})
_FAILED_STATUSES = frozenset({"failed", "error"})


class _WorkerNetError(Exception):
    """Сбой связи с воркером (без деталей: они могут содержать заголовки)."""


class _WorkerHttpError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


@dataclass
class RunSummary:
    total_needed: int = 0      # сколько клипов нужно было сделать в этом запуске
    completed: int = 0
    failed: int = 0
    pending: int = 0
    skipped_done: int = 0      # уже лежали в Release
    spent_usd: float = 0.0
    exit_code: int = EXIT_OK
    deadline_reached: bool = False
    budget_exceeded: bool = False
    cleanup_ok: bool = True
    cards_rented: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Task:
    num: int
    id: str
    started: float


@dataclass
class _CardSlot:
    index: int
    generation: int
    instance_id: int
    machine_id: int
    price: float
    created: float
    state: str = "booting"     # booting -> ready
    url: str = ""
    contacted: bool = False    # воркер хотя бы раз ответил
    net_fail: int = 0
    task: Optional[_Task] = None


def _prompt_ready(entry) -> bool:
    return (isinstance(entry, dict) and entry.get("status") == "ready"
            and bool(str(entry.get("prompt", "")).strip()))


class Orchestrator:
    """Координатор: промпты, карты Vast, рендер, выдача в Release, уборка."""

    def __init__(
        self,
        segments: list,
        missing_path,
        cfg,
        timing_params,
        release_store,
        vast_client,
        gemini_transport,
        prompts_path,
        *,
        max_cards: int = 4,
        budget_limit_usd: float = 5.0,
        job_deadline_min: int = 330,
        card_max_lifetime_min: int = 90,
        silent_host_timeout_min: int = 5,
        worker_idle_timeout_min: int = 5,
        offer_filter: Optional[OfferFilter] = None,
        docker_image: str = "",
        disk_gb: int = 40,
        limit_clips: int = 0,
        repo: str = "",
        run_id: str = "",
        profile: str = "",
        filename_template: str = "{num}.mp4",
        clip_name_width: int = 0,
        release_tag: Optional[str] = None,
        release_asset_limit: int = 900,
        gemini_options: Optional[dict] = None,
        http_client: Optional[httpx.Client] = None,
        work_dir=None,
        worker_token: Optional[str] = None,
        seed_base: int = 0,
        poll_interval_sec: float = 10.0,
        rent_retry_sec: float = 30.0,
        drain_grace_sec: float = 900.0,
        task_timeout_sec: float = 1200.0,
        render_estimate_sec: float = 120.0,
        rent_estimate_sec: float = 1800.0,
        http_timeout_sec: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_cards <= 0:
            raise gp.GeminiInputError("max_cards должен быть больше нуля.")
        if job_deadline_min <= 0 or card_max_lifetime_min <= 0 or silent_host_timeout_min <= 0:
            raise gp.GeminiInputError("Сроки (дедлайн, жизнь карты, молчание хоста) должны быть положительными.")
        self.segments = segments
        self.missing_path = missing_path
        self.cfg = cfg
        self.timing_params = timing_params
        self.release_store = release_store
        self.vast = vast_client
        self.gemini_transport = gemini_transport
        self.prompts_path = Path(prompts_path)

        self.max_cards = int(max_cards)
        self.job_deadline_min = job_deadline_min
        self.worker_idle_timeout_min = worker_idle_timeout_min
        self._deadline_sec = job_deadline_min * 60.0
        self._lifetime_sec = card_max_lifetime_min * 60.0
        self._silent_sec = silent_host_timeout_min * 60.0
        self.card_max_lifetime_min = card_max_lifetime_min
        self.offer_filter = offer_filter or OfferFilter()
        self.docker_image = docker_image
        self.disk_gb = disk_gb
        self.limit_clips = limit_clips
        self.repo = repo
        self.label = f"gen-{run_id or uuid.uuid4().hex[:8]}"
        self.profile = profile
        self.template = filename_template
        self.width = clip_name_width
        self.release_tag = release_tag
        self.release_asset_limit = release_asset_limit
        self.gemini_options = dict(gemini_options or {})
        self.work_dir = work_dir
        self.seed_base = seed_base
        self.poll_interval = poll_interval_sec
        self.rent_retry = rent_retry_sec
        self.drain_grace = drain_grace_sec
        self.task_timeout = task_timeout_sec
        self.render_estimate = render_estimate_sec
        self.rent_estimate = rent_estimate_sec
        self.http_timeout = http_timeout_sec
        self.clock = clock
        self._sleep = sleep

        self._token = worker_token or secrets.token_urlsafe(24)
        self._headers = {"X-Worker-Token": self._token}
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.Client(timeout=http_timeout_sec)
        self.budget = BudgetTracker(budget_limit_usd, clock=clock)

        # состояние запуска
        self._t0 = 0.0
        self._tmp = ""
        self._uploader: Optional[ReleaseUploader] = None
        self._state: dict = {"prompts": {}}
        self._prompts: dict = self._state["prompts"]
        self._queue: deque = deque()
        self._slots: dict = {}
        self._gens: dict = {}
        self._attempts: dict = {}
        self._done: set = set()
        self._failed: set = set()
        self._durations: list = []
        self._bad_machines: set = set()
        self._next_rent = 0.0
        self._budget_blocked = False
        self._cards_rented = 0
        self.deadline_reached = False
        self.budget_exceeded = False
        self.cleanup_ok = True

    def __repr__(self) -> str:
        return f"Orchestrator(метка={self.label}, карт<={self.max_cards})"

    __str__ = __repr__

    # --- служебное ---------------------------------------------------------
    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    def _info(self, msg: str) -> None:
        log.info("%s", msg)

    # --- запуск -------------------------------------------------------------
    def run(self) -> RunSummary:
        self._t0 = self.clock()
        try:
            missing = parse_missing(self.missing_path, self.segments)
            if not missing:
                log.info("Пропущенных сегментов нет: генерировать нечего.")
                return RunSummary()
            if self.limit_clips > 0:
                missing = missing[: self.limit_clips]
                log.info("Пробный прогон: берём первые %d номеров.", len(missing))
            if self.work_dir:
                Path(self.work_dir).mkdir(parents=True, exist_ok=True)
            self._tmp = tempfile.mkdtemp(prefix="gen-", dir=str(self.work_dir) if self.work_dir else None)
            return self._run_inner(missing)
        finally:
            if self._tmp:
                shutil.rmtree(self._tmp, ignore_errors=True)
            if self._owns_http:
                self._http.close()

    def _run_inner(self, missing: list) -> RunSummary:
        uploader = ReleaseUploader(
            self.release_store, self.segments, self.profile, release_tag=self.release_tag,
            release_asset_limit=self.release_asset_limit, filename_template=self.template,
            log=self._info)
        done = uploader.prepare()
        self._uploader = uploader
        needed = [n for n in missing if n not in done]
        summary = RunSummary(total_needed=len(needed), skipped_done=len(missing) - len(needed))
        if not needed:
            log.info("Все запрошенные клипы уже в Release (%d): аренда карт не требуется.", len(missing))
            self._upload_services()
            return summary

        try:
            ready = self._generate(needed)
            self._work(ready)
        finally:
            self._cleanup()
            self._upload_services()

        summary.completed = len(self._done)
        summary.failed = len(self._failed)
        summary.pending = len(needed) - summary.completed - summary.failed
        summary.spent_usd = self.budget.current_spent(self.clock())
        summary.deadline_reached = self.deadline_reached
        summary.budget_exceeded = self.budget_exceeded
        summary.cleanup_ok = self.cleanup_ok
        summary.cards_rented = self._cards_rented
        full = summary.completed == len(needed) and self.cleanup_ok
        summary.exit_code = EXIT_OK if full else EXIT_PARTIAL
        log.info("Итог: готово %d из %d, ошибок %d, не начато %d, потрачено %.4f USD, код %d.",
                 summary.completed, len(needed), summary.failed, summary.pending,
                 summary.spent_usd, summary.exit_code)
        return summary

    # --- промпты -------------------------------------------------------------
    def _read_state(self) -> dict:
        try:
            state = json.loads(self.prompts_path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and isinstance(state.get("prompts"), dict):
                return state
        except (OSError, ValueError):
            pass
        return {"prompts": {}}

    def _generate(self, needed: list) -> list:
        try:
            state = go.generate_and_store(
                self.segments, needed, self.cfg, self.gemini_transport, self.prompts_path,
                profile=self.profile, timing=self.timing_params, **self.gemini_options)
        except gp.GeminiRuntimeError as exc:
            log.error("Генерация промптов прервана: %s. Используем уже сохранённые.", exc)
            state = self._read_state()
        self._state = state
        self._prompts = state.setdefault("prompts", {})
        ready = [n for n in needed if _prompt_ready(self._prompts.get(str(n)))]
        for n in needed:
            if n not in ready:
                self._failed.add(n)
                self._mark(n, "failed")
        if len(ready) < len(needed):
            log.warning("Без готового промпта осталось клипов: %d.", len(needed) - len(ready))
        return ready

    def _save_state(self) -> None:
        try:
            go.atomic_write_json(self.prompts_path, self._state)
        except OSError:
            log.warning("Не удалось сохранить файл промптов.")

    def _mark(self, num: int, status: str) -> None:
        entry = self._prompts.get(str(num))
        if isinstance(entry, dict):
            entry["render_status"] = status
            self._save_state()

    # --- основной цикл -------------------------------------------------------
    def _inflight(self) -> int:
        return sum(1 for s in self._slots.values() if s.task is not None)

    def _work(self, ready: list) -> None:
        self._queue = deque(ready)
        drain_since: Optional[float] = None
        while True:
            now = self.clock()
            deadline_hit = (now - self._t0) >= self._deadline_sec
            if deadline_hit and drain_since is None:
                drain_since = now
                self.deadline_reached = True
                log.warning("Достигнут срок job (%d мин): новые клипы не берём, дозаливаем начатое.",
                            self.job_deadline_min)
            if not self.budget_exceeded and self.budget.is_exceeded(now):
                self.budget_exceeded = True
                log.error("Бюджет исчерпан: снимаем все карты, новые не арендуем.")
                for slot in list(self._slots.values()):
                    self._retire(slot, "бюджет исчерпан")
            stopping = deadline_hit or self.budget_exceeded

            for slot in list(self._slots.values()):
                self._step_slot(slot, now, stopping)

            inflight = self._inflight()
            if stopping:
                if inflight == 0:
                    break
                if drain_since is not None and now - drain_since >= self.drain_grace:
                    log.error("Время дозаливки вышло: незавершённых клипов %d.", inflight)
                    break
            else:
                if not self._queue and inflight == 0:
                    break
                self._maybe_rent(now)
                if self._budget_blocked and not self._slots:
                    log.error("Бюджет не позволяет арендовать карту, активных карт нет: завершаем.")
                    break
            self._sleep(self.poll_interval)

    # --- аренда --------------------------------------------------------------
    def _desired(self, now: float) -> int:
        remaining = len(self._queue) + self._inflight()
        if remaining <= 0:
            return 0
        avg = sum(self._durations) / len(self._durations) if self._durations else self.render_estimate
        time_left = max(1.0, self._deadline_sec - (now - self._t0))
        need = math.ceil(remaining * avg / (time_left * 0.8))
        return max(1, min(self.max_cards, need, remaining))

    def _maybe_rent(self, now: float) -> None:
        if self._budget_blocked or now < self._next_rent:
            return
        while len(self._slots) < self._desired(now):
            res = self._rent_one(now)
            if res == "retry":
                self._next_rent = now + self.rent_retry
                return
            if res == "budget":
                self._budget_blocked = True
                return

    def _rent_one(self, now: float) -> str:
        try:
            offers = self.vast.search_offers(self.offer_filter)
        except VastError as exc:
            log.warning("Поиск предложений Vast не удался: %s", exc)
            return "retry"
        offers = [o for o in offers if not (o.machine_id and o.machine_id in self._bad_machines)]
        if not offers:
            log.warning("Подходящих предложений Vast нет, повтор позже.")
            return "retry"
        est = min(self.rent_estimate, self._lifetime_sec)
        affordable = [o for o in offers if self.budget.can_afford(o.price_per_hr, est, now)]
        if not affordable:
            log.warning("Бюджета не хватает ни на одно предложение (остаток %.4f USD).",
                        self.budget.remaining_budget(now))
            return "budget"
        offer = affordable[0]
        env = {
            "WORKER_TOKEN": self._token,
            "WORKER_IDLE_TIMEOUT_MIN": str(self.worker_idle_timeout_min),
            "WORKER_MAX_LIFETIME_MIN": str(self.card_max_lifetime_min),
            f"-p {WORKER_HTTP_PORT}:{WORKER_HTTP_PORT}": "1",
        }
        try:
            inst_id = self.vast.create_instance(
                offer.offer_id, self.docker_image, self.disk_gb, env_vars=env, label=self.label)
        except VastError as exc:
            log.warning("Не удалось создать карту по предложению %d: %s", offer.offer_id, exc)
            return "retry"
        self.budget.track_instance_start(inst_id, offer.price_per_hr, now)
        index = next(i for i in range(self.max_cards) if i not in self._slots)
        gen = self._gens.get(index, 0) + 1
        self._gens[index] = gen
        self._slots[index] = _CardSlot(index, gen, inst_id, offer.machine_id,
                                       offer.price_per_hr, now)
        self._cards_rented += 1
        log.info("Арендована карта %d (слот %d, поколение %d): %.4f USD/ч.",
                 inst_id, index, gen, offer.price_per_hr)
        return "ok"

    def _retire(self, slot: _CardSlot, reason: str, *, bad: bool = False) -> None:
        log.warning("Карта %d (слот %d, поколение %d) снимается: %s.",
                    slot.instance_id, slot.index, slot.generation, reason)
        if slot.task is not None:
            self._requeue_free(slot.task.num)
            slot.task = None
        if bad and slot.machine_id:
            self._bad_machines.add(slot.machine_id)
        try:
            self.vast.destroy_instance(slot.instance_id)
        except VastError as exc:
            log.error("Не удалось уничтожить карту %d: %s. Будет повтор при финальной уборке.",
                      slot.instance_id, exc)
        else:
            try:
                self.budget.track_instance_stop(slot.instance_id, self.clock())
            except BudgetError:
                pass
        self._slots.pop(slot.index, None)

    # --- шаг по карте ---------------------------------------------------------
    def _step_slot(self, slot: _CardSlot, now: float, stopping: bool) -> None:
        age = now - slot.created
        if slot.state == "booting":
            info = None
            try:
                info = self.vast.get_instance(slot.instance_id)
            except VastError as exc:
                log.warning("Не удалось получить статус карты %d: %s", slot.instance_id, exc)
            if info is not None:
                if info.actual_status == "exited" or info.intended_status == "stopped":
                    self._retire(slot, "карта остановлена хостом", bad=True)
                    return
                if info.actual_status == "running" and info.public_ip and info.direct_port:
                    slot.url = f"http://{info.public_ip}:{info.direct_port}"
                    slot.state = "ready"
                    log.info("Карта %d запущена, воркер доступен по прямому порту.", slot.instance_id)
        if not slot.contacted and age >= self._silent_sec:
            self._retire(slot, f"хост не вышел на связь за {int(self._silent_sec // 60)} мин", bad=True)
            return
        if age >= self._lifetime_sec:
            self._retire(slot, "истёк предельный срок жизни карты")
            return
        if slot.net_fail >= MAX_NET_FAILS:
            self._retire(slot, "воркер недоступен", bad=True)
            return
        if slot.state != "ready":
            return
        if slot.task is not None:
            self._poll_task(slot, now)
        if self._slots.get(slot.index) is not slot or slot.task is not None:
            return
        if not stopping and self._queue:
            self._dispatch(slot, now)
        elif stopping or not self._queue:
            self._retire(slot, "работы для карты больше нет")

    # --- воркер ----------------------------------------------------------------
    def _call(self, slot: _CardSlot, method: str, path: str, payload: Optional[dict] = None):
        try:
            resp = self._http.request(method, slot.url + path, json=payload,
                                      headers=self._headers, timeout=self.http_timeout)
        except httpx.HTTPError as exc:
            raise _WorkerNetError(type(exc).__name__) from None
        if resp.status_code >= 400:
            raise _WorkerHttpError(resp.status_code)
        return resp

    @staticmethod
    def _json(resp) -> dict:
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _requeue_free(self, num: int) -> None:
        if self._attempts.get(num, 0) > 0:
            self._attempts[num] -= 1
        self._queue.appendleft(num)

    def _fail_clip(self, num: int, reason: str) -> None:
        used = self._attempts.get(num, 0)
        if used >= MAX_ATTEMPTS:
            log.error("Клип %d: попытки исчерпаны (%d), помечен как неудачный. Причина: %s",
                      num, used, reason)
            self._failed.add(num)
            self._mark(num, "failed")
        else:
            log.warning("Клип %d: неудачная попытка %d из %d (%s), возвращён в очередь.",
                        num, used, MAX_ATTEMPTS, reason)
            self._queue.append(num)

    def _dispatch(self, slot: _CardSlot, now: float) -> None:
        num = self._queue.popleft()
        entry = self._prompts[str(num)]
        payload = {"num": num, "prompt": entry["prompt"],
                   "num_frames": entry.get("num_frames"), "seed": self.seed_base + num}
        try:
            resp = self._call(slot, "POST", "/task", payload)
        except _WorkerNetError as exc:
            slot.net_fail += 1
            self._queue.appendleft(num)
            log.warning("Воркер карты %d не принял клип %d (сбой связи: %s).",
                        slot.instance_id, num, exc)
            return
        except _WorkerHttpError as exc:
            slot.net_fail += 1
            self._attempts[num] = self._attempts.get(num, 0) + 1
            self._fail_clip(num, f"воркер ответил {exc}")
            return
        slot.contacted = True
        slot.net_fail = 0
        self._attempts[num] = self._attempts.get(num, 0) + 1
        data = self._json(resp)
        tid = str(data.get("id") or data.get("task_id") or num)
        slot.task = _Task(num, tid, now)
        log.info("Клип %d отправлен на карту %d (попытка %d).",
                 num, slot.instance_id, self._attempts[num])

    def _poll_task(self, slot: _CardSlot, now: float) -> None:
        task = slot.task
        if now - task.started >= self.task_timeout:
            slot.task = None
            self._fail_clip(task.num, "таймаут рендера")
            self._retire(slot, "воркер завис", bad=True)
            return
        try:
            resp = self._call(slot, "GET", f"/task/{task.id}")
        except _WorkerNetError:
            slot.net_fail += 1
            return
        except _WorkerHttpError as exc:
            slot.task = None
            self._fail_clip(task.num, f"статус задачи: {exc}")
            return
        slot.net_fail = 0
        slot.contacted = True
        data = self._json(resp)
        status = str(data.get("status") or data.get("state") or "").lower()
        if status in _DONE_STATUSES:
            self._finish_clip(slot, now)
        elif status in _FAILED_STATUSES:
            slot.task = None
            err = self._redact(str(data.get("error", "")))[:200]
            self._fail_clip(task.num, f"воркер сообщил об ошибке: {err}")

    def _download(self, slot: _CardSlot, tid: str, path: str) -> None:
        try:
            with self._http.stream("GET", f"{slot.url}/file/{tid}", headers=self._headers,
                                   timeout=self.http_timeout) as resp:
                if resp.status_code >= 400:
                    raise _WorkerHttpError(resp.status_code)
                with open(path, "wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
        except httpx.HTTPError as exc:
            raise _WorkerNetError(type(exc).__name__) from None

    def _finish_clip(self, slot: _CardSlot, now: float) -> None:
        task = slot.task
        path = os.path.join(self._tmp, clip_filename(task.num, self.width, self.template))
        reason = ""
        try:
            self._download(slot, task.id, path)
            self._uploader.upload_clip(path)
        except (_WorkerNetError, _WorkerHttpError) as exc:
            reason = f"не удалось скачать клип с воркера ({exc})"
        except gp.GeminiRuntimeError as exc:
            reason = f"не удалось загрузить клип в Release: {exc}"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        slot.task = None
        if reason:
            self._fail_clip(task.num, reason)
            return
        self._done.add(task.num)
        self._durations.append(max(0.0, now - task.started))
        self._mark(task.num, "done")
        log.info("Клип %d выдан в Release.", task.num)
        try:
            self._call(slot, "POST", f"/ack/{task.id}")
        except (_WorkerNetError, _WorkerHttpError):
            log.warning("Воркер не подтвердил удаление клипа %d с диска карты.", task.num)

    # --- финал -------------------------------------------------------------------
    def _cleanup(self) -> None:
        try:
            ids = self.vast.destroy_by_label(self.label)
            log.info("Уничтожено карт по метке %s: %d.", self.label, len(ids))
        except Exception as exc:  # уборка не должна маскировать исходную ошибку
            self.cleanup_ok = False
            log.error("Не удалось уничтожить карты по метке %s: %s", self.label,
                      exc if isinstance(exc, VastError) else type(exc).__name__)
            return
        now = self.clock()
        for rec in self.budget.summary(now)["breakdown"]:
            if rec["active"]:
                try:
                    self.budget.track_instance_stop(rec["instance_id"], now)
                except BudgetError:
                    pass

    def _upload_services(self) -> None:
        """prompts.json и generated_links.txt в основной релиз; ошибки не фатальны."""
        if self._uploader is None or not self._tmp:
            return
        try:
            if self.prompts_path.exists():
                dst = os.path.join(self._tmp, PROMPTS_FILENAME)
                shutil.copyfile(self.prompts_path, dst)
                self._uploader.upload_service_file(dst)
            if self.repo:
                dst = os.path.join(self._tmp, LINKS_FILENAME)
                Path(dst).write_text(self._uploader.links_text(self.repo), encoding="utf-8")
                self._uploader.upload_service_file(dst)
            else:
                log.warning("Репозиторий не задан: generated_links.txt не создан.")
        except Exception as exc:
            log.error("Не удалось выложить служебные файлы: %s",
                      exc if isinstance(exc, gp.GeminiRuntimeError) else type(exc).__name__)


def run_generation(*args, **kwargs) -> RunSummary:
    """Удобная обёртка: создаёт Orchestrator и запускает конвейер."""
    return Orchestrator(*args, **kwargs).run()
