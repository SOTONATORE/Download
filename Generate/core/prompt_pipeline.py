"""Общий конвейер промптов для видео и фото (STEP 7, блок B2; вынесено из orchestrator.py).

Единственное место, где готовятся промпты. Генерация (Orchestrator._generate) и проверка
(cli.prepare_prompts_only, workflow verify_prompts) вызывают build_prompts с одними и теми же входами.
Проверки запрещённых типов кадров и незаполненных маркеров, а также переписывание (rework loop)
выполняются внутри gemini_prompts.generate_prompts (шаг 6), поэтому идут по этому пути автоматически.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from . import gemini_orchestrator as go
    from . import gemini_prompts as gp
except ImportError:  # pragma: no cover
    import gemini_orchestrator as go  # type: ignore
    import gemini_prompts as gp  # type: ignore

# тот же логгер, что у оркестратора: строки журнала не меняются
log = logging.getLogger("Generate.orchestrator")

PHOTO_NUM_FRAMES = 1        # фото: один промпт = один кадр; задаётся кодом, не из тайминга
VIDEO_WAVE = "видео"        # имя волны видео в общем пуле батчей
PHOTO_WAVE = "фото"         # имя волны фото в общем пуле батчей


@dataclass
class PromptRun:
    """Итог подготовки промптов.

    ready: номера из needed, у которых есть готовый промпт, в порядке needed;
    state: содержимое prompts.json (словарь с ключом prompts);
    errors: тексты остановок шагов Gemini (GeminiRuntimeError), которые были перехвачены и записаны в журнал;
    abort: None, если жёсткой ошибки не было; иначе «волна, сегмент N» (N — «н/д», если номер не известен).
    При abort оркестратор не арендует карты и не запускает рендер.
    """
    ready: list
    state: dict
    errors: list = field(default_factory=list)
    abort: Optional[str] = None


def _prompt_ready(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    num_frames = entry.get("num_frames")
    return (entry.get("status") == "ready"
            and bool(str(entry.get("prompt", "")).strip())
            and isinstance(num_frames, int) and not isinstance(num_frames, bool)
            and num_frames > 0)


def read_state(prompts_path) -> dict:
    """Читает prompts.json; при ошибке чтения или формата возвращает пустое состояние."""
    try:
        state = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
        if isinstance(state, dict) and isinstance(state.get("prompts"), dict):
            return state
    except (OSError, ValueError):
        pass
    return {"prompts": {}}


class _TransportWatch:
    """Обёртка над транспортом Gemini: помнит исход вызовов, чтобы отличить временный сбой от жёсткой ошибки.

    Повторы временных сбоев живут внутри ManagedTransport. Если они исчерпаны, последний вызов завершился
    временным сбоем (last_transient=True). Ошибка переписывания (запрещённые кадры, маркеры) приходит
    после успешных вызовов: calls > 0 и last_transient=False. Число повторов и переписываний не меняется.

    При общем пуле волны идут параллельно, поэтому счётчики ведутся и по волнам: поток пула перед батчем
    вызывает bind(волна), и вызовы этого потока учитываются в её счётчиках (см. view). Общие calls и
    last_transient остаются как были.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._lock = threading.Lock()   # счётчики меняются из потоков пула
        self._local = threading.local()  # волна батча, который выполняет текущий поток
        self.calls = 0                  # сколько раз вызывали транспорт (всего)
        self.last_transient = False     # последний вызов завершился временным сбоем (всего)
        self._wave_calls: dict = {}     # волна -> число вызовов
        self._wave_transient: dict = {}  # волна -> последний вызов завершился временным сбоем

    def bind(self, wave) -> None:
        """Вызывается пулом в потоке батча: вызовы этого потока относятся к волне wave (None — без учёта)."""
        self._local.wave = wave

    def view(self, wave) -> "_WaveView":
        """Счётчики одной волны для _is_hard."""
        return _WaveView(self, wave)

    def generate_json(self, model, system, user, schema):
        wave = getattr(self._local, "wave", None)
        with self._lock:
            self.calls += 1
            if wave is not None:
                self._wave_calls[wave] = self._wave_calls.get(wave, 0) + 1
        try:
            text = self._inner.generate_json(model, system, user, schema)
        except gp.GeminiRuntimeError as exc:
            self._record(wave, go.is_transient(exc))
            raise
        self._record(wave, False)
        return text

    def _record(self, wave, transient: bool) -> None:
        with self._lock:
            self.last_transient = transient
            if wave is not None:
                self._wave_transient[wave] = transient

    def __getattr__(self, name):  # прочие атрибуты (например, active_model) берём у внутреннего транспорта
        return getattr(self._inner, name)


class _WaveView:
    """Счётчики одной волны: calls и last_transient только для вызовов этой волны."""

    def __init__(self, watch: "_TransportWatch", wave: str) -> None:
        self._watch = watch
        self._wave = wave

    @property
    def calls(self) -> int:
        return self._watch._wave_calls.get(self._wave, 0)

    @property
    def last_transient(self) -> bool:
        return self._watch._wave_transient.get(self._wave, False)


def _is_hard(exc, watch: "_TransportWatch", mark: int = 0) -> bool:
    """Жёсткая ошибка: любой сбой, кроме временного, и ошибка шага после успешных вызовов.

    mark: число вызовов транспорта на начало волны; учитываются только вызовы этой волны.
    Временный сбой (is_transient) после исчерпания повторов или без вызовов волны остаётся
    возвратом к сохранённым промптам, как раньше.
    Ошибка шага без статуса (status None) для is_transient выглядит как сетевой сбой, поэтому
    решает счётчик вызовов: успешный вызов в этой волне и последний вызов не временный.
    """
    if not go.is_transient(exc):
        return True
    return watch.calls > mark and not watch.last_transient


def _abort_text(wave: str, exc) -> str:
    """«волна, сегмент N» для строки об остановке; номер берётся из текста ошибки шага (Сегмент N:)."""
    m = re.search(r"Сегмент (\d+):", str(exc))
    return f"{wave}, сегмент {m.group(1) if m else 'н/д'}"


def _video_step(needed: list, *, segments, cfg, watch, prompts_path, profile, timing,
                gemini_options: dict, errors: list, aborts: list,
                managed=None, state=None, lock=None, pool=None, progress=None) -> dict:
    """Промпты для видео: generate_and_store дописывает prompts.json после каждого батча.

    watch: общий наблюдатель прогона (через него идут все вызовы, включая цепочку managed).
    managed, state, lock: общие цепочка, состояние и замок прогона; None — generate_and_store создаёт их сам.
    pool: общий пул батчей обеих волн; None — прежний путь одной волны.
    progress: строка прогресса обеих волн (только при пуле); None — строк прогресса нет.
    """
    # при пуле ошибка классифицируется по вызовам этой волны; без пула — по общим счётчикам, как раньше
    seen = watch.view(VIDEO_WAVE) if pool is not None else watch
    mark = 0 if pool is not None else watch.calls  # вызовы этой волны начинаются отсюда
    pool_kw = {} if pool is None else {"pool": pool, "wave": VIDEO_WAVE, "progress": progress}
    try:
        return go.generate_and_store(
            segments, needed, cfg, watch, prompts_path,
            profile=profile, timing=timing, managed=managed, state=state, lock=lock, **pool_kw, **gemini_options)
    except gp.GeminiRuntimeError as exc:
        errors.append(str(exc))
        if _is_hard(exc, seen, mark):
            aborts.append(_abort_text("видео", exc))
            log.error("Генерация промптов (видео) остановлена: %s.", exc)
            if pool is not None:
                pool.fail(exc)  # вторая волна доберёт текущие батчи, сохранит их и не возьмёт новых
            return state if state is not None else read_state(prompts_path)
        log.error("Генерация промптов прервана: %s. Используем уже сохранённые.", exc)
        return state if state is not None else read_state(prompts_path)
    finally:
        if pool is not None:
            pool.leave(VIDEO_WAVE)  # волна вышла из пула (даже без батчей): фото не ждёт её вечно


def _photo_step(needed: list, *, segments, photo_cfg, watch, prompts_path,
                gemini_options: dict, errors: list, aborts: list,
                managed, state: dict, lock, pool=None, progress=None) -> None:
    """Промпты для фото: без тайминга; num_frames всегда 1 (STEP 7, блок B2).

    watch: общий наблюдатель прогона; managed: общая цепочка, построенная на нём;
    state, lock: общие состояние и замок прогона. Профиль фото в состояние не пишется:
    поле profile файла задаёт build_prompts до волн и волна его не перезаписывает.
    pool: общий пул батчей обеих волн; None — прежний путь одной волны.
    progress: строка прогресса обеих волн (только при пуле); None — строк прогресса нет.
    """
    seen = watch.view(PHOTO_WAVE) if pool is not None else watch
    mark = 0 if pool is not None else watch.calls  # вызовы этой волны начинаются отсюда
    try:
        guard = lock
        prompts = state["prompts"]
        pending = go.missing_segments(prompts, needed)
        if not pending:
            return

        def on_batch(results: list) -> None:
            # под замком: видео и фото пишут один общий state, слияние и запись не должны перемешиваться
            with guard:
                for r in results:
                    old = prompts.get(str(r.num)) or {}
                    entry = {
                        "num": r.num, "num_frames": PHOTO_NUM_FRAMES, "prompt": r.prompt,
                        "status": "ready" if r.prompt.strip() else "failed",
                        "attempts": int(old.get("attempts", 0)) + 1,
                        "error": r.review_reason, "needs_review": r.needs_review, "extra": r.extra,
                    }
                    rework_round = getattr(r, "rework_round", None)
                    if rework_round is not None:
                        # поля очереди исправлений — только у клипов, которые её проходили
                        entry["rework_pending"] = list(getattr(r, "rework_pending", None) or [])
                        entry["rework_round"] = int(rework_round)
                    prompts[str(r.num)] = entry
                state["model"] = managed.active_model
                state["prompts"] = {k: prompts[k] for k in sorted(prompts, key=int)}
                go.atomic_write_json(prompts_path, state)
                if progress is not None:
                    # прогресс обеих волн: из того же словаря, под тем же замком, уже после записи файла
                    log.info(progress(prompts), extra=gp.KEY_EVENT_EXTRA)

        # провайдер у транспорта (через _TransportWatch); для gemini аргумент не передаётся, вызов как раньше
        provider = getattr(watch, "provider", "gemini")
        extra = {} if provider == "gemini" else {"provider": provider}
        pool_kw = {} if pool is None else {"pool": pool, "wave": PHOTO_WAVE}
        gp.generate_prompts(segments, pending, photo_cfg, managed, managed.active_model, on_batch=on_batch,
                            **extra, **pool_kw)
    except gp.GeminiRuntimeError as exc:
        errors.append(str(exc))
        if _is_hard(exc, seen, mark):
            aborts.append(_abort_text("фото", exc))
            log.error("Генерация промптов для фото остановлена: %s.", exc)
            if pool is not None:
                pool.fail(exc)  # вторая волна доберёт текущие батчи, сохранит их и не возьмёт новых
            return
        log.error("Генерация промптов для фото прервана: %s. Используем уже сохранённые.", exc)
    finally:
        if pool is not None:
            pool.leave(PHOTO_WAVE)  # волна вышла из пула (даже без батчей): видео не ждёт её вечно


def _load_shared(state, prompts_path, segments, profile) -> dict:
    """Общий словарь состояния прогона: читается из файла один раз, при первом запуске волны."""
    if state is None:
        return go.load_state(prompts_path, segments, profile)
    return state


def _pool_threads(provider: str) -> int:
    """Всего потоков общего пула: число ключей провайдера, не меньше 1 и не больше MAX_PARALLEL_BATCHES."""
    try:
        keys = gp._provider_key_count(provider)
    except Exception:  # ключей нет или не прочитались: как в generate_prompts, один поток
        keys = 0
    return min(max(keys, 1), gp.MAX_PARALLEL_BATCHES)


def _progress_for(video: list, photo: list, started: float):
    """Функция строки прогресса для колбэков сохранения пула: читает словарь prompts и вызывается под замком состояния."""
    def progress(prompts: dict) -> str:
        return go.progress_line(prompts, video, photo, started)
    return progress


def _run_waves_together(video: list, photo: list, *, segments, cfg, photo_cfg, watch, managed,
                        prompts_path, profile, timing, gemini_options: dict, errors: list, aborts: list,
                        state: dict, lock, provider: str, progress=None) -> dict:
    """Видео и фото одновременно через общий пул батчей. Потоки делятся по ключам: видео floor(K/2), фото ceil(K/2).

    Фото идёт в отдельном потоке, видео в вызывающем. Обе волны получают общие state и lock. Оба читают
    state до барьера пула, поэтому пишут в один словарь prompts. Ошибка видео вне батчей останавливает пул.
    Ошибка фото пробрасывается после обеих волн; видео имеет приоритет, если упали обе.
    """
    total = _pool_threads(provider)
    video_threads = total // 2
    pool = gp.SharedBatchPool({VIDEO_WAVE: video_threads, PHOTO_WAVE: total - video_threads}, bind=watch.bind)
    box: dict = {}

    def run_photo() -> None:
        try:
            _photo_step(photo, segments=segments, photo_cfg=photo_cfg, watch=watch, prompts_path=prompts_path,
                        gemini_options=gemini_options, errors=errors, aborts=aborts,
                        managed=managed, state=state, lock=lock, pool=pool, progress=progress)
        except BaseException as exc:  # пробрасывается в основной поток после обеих волн
            box["photo"] = exc

    th = threading.Thread(target=run_photo, name="photo-wave")
    th.start()
    try:
        state = _video_step(video, segments=segments, cfg=cfg, watch=watch,
                            prompts_path=prompts_path, profile=profile, timing=timing,
                            gemini_options=gemini_options, errors=errors, aborts=aborts,
                            managed=managed, state=state, lock=lock, pool=pool, progress=progress)
    except BaseException as exc:
        box["video"] = exc
        pool.fail(exc)   # видео упало вне батчей: фото новые батчи не берёт, как раньше не запускался бы вовсе
    th.join()
    if "video" in box:
        raise box["video"]
    if "photo" in box:
        raise box["photo"]
    return state


def build_prompts(needed: list, *, segments: list, photo_nums, cfg, profile: str, timing,
                  photo_profile: Optional[str], photo_cfg, transport, gemini_options: dict,
                  prompts_path) -> PromptRun:
    """Единый конвейер промптов для списка номеров needed.

    photo_profile None: все номера идут в видео (photo_nums не читается); состояние и цепочку
    создаёт generate_and_store, фото-путь не вызывается.
    Иначе номера из photo_nums идут в фото, остальные в видео. Если есть оба вида, волны идут
    параллельно через общий пул батчей; если один — по одной волне, как раньше. Затем состояние
    читается из файла (оно содержит оба вида).
    Поле profile в файле одно: профиль видео, если видео есть в прогоне, иначе профиль фото.
    Значение зависит только от состава прогона, а не от порядка волн.
    """
    started = time.monotonic()  # начало прогона: от него считается «за T с» в строке прогресса
    errors: list = []
    aborts: list = []  # жёсткие ошибки волн; решение об остановке принимает оркестратор после обеих волн
    # общие объекты прогона: один наблюдатель транспорта, один замок и одна цепочка моделей на оба вида
    watch = _TransportWatch(transport)  # все вызовы Gemini идут через него, и видео, и фото
    lock = threading.Lock()
    state = None
    if photo_profile is None:
        state = _video_step(list(needed), segments=segments, cfg=cfg, watch=watch,
                            prompts_path=prompts_path, profile=profile, timing=timing,
                            gemini_options=gemini_options, errors=errors, aborts=aborts,
                            managed=None, state=None, lock=lock)
    else:
        video = [n for n in needed if n not in photo_nums]
        photo = [n for n in needed if n in photo_nums]
        # цепочка строится на наблюдателе: её вызовы видны _TransportWatch и для видео, и для фото
        managed = go.ManagedTransport(
            watch, gemini_options.get("models", go.DEFAULT_MODELS),
            rpm_limit=gemini_options.get("rpm_limit", 15), tpm_limit=gemini_options.get("tpm_limit", 250_000),
            rpd_limit=gemini_options.get("rpd_limit", 500))
        # профиль файла: видео, если в прогоне есть видео; иначе фото (решается до волн, не по порядку)
        file_profile = profile if video else photo_profile
        if video and photo:
            # оба вида: параллельно через общий пул; состояние читается один раз до старта потоков
            state = _load_shared(state, prompts_path, segments, file_profile)
            state = _run_waves_together(video, photo, segments=segments, cfg=cfg, photo_cfg=photo_cfg,
                                        watch=watch, managed=managed, prompts_path=prompts_path,
                                        profile=profile, timing=timing, gemini_options=gemini_options,
                                        errors=errors, aborts=aborts, state=state, lock=lock,
                                        provider=getattr(watch, "provider", "gemini"),
                                        progress=_progress_for(video, photo, started))
        else:
            if video:
                state = _load_shared(state, prompts_path, segments, file_profile)
                state = _video_step(video, segments=segments, cfg=cfg, watch=watch,
                                    prompts_path=prompts_path, profile=profile, timing=timing,
                                    gemini_options=gemini_options, errors=errors, aborts=aborts,
                                    managed=managed, state=state, lock=lock)
            if photo:
                state = _load_shared(state, prompts_path, segments, file_profile)
                _photo_step(photo, segments=segments, photo_cfg=photo_cfg, watch=watch,
                            prompts_path=prompts_path, gemini_options=gemini_options,
                            errors=errors, aborts=aborts, managed=managed, state=state, lock=lock)
        state = read_state(prompts_path)  # после обеих волн: состояние файла с обоими видами
    prompts = state.setdefault("prompts", {})
    ready = [n for n in needed if _prompt_ready(prompts.get(str(n)))]
    if aborts:
        # итог до остановки: сколько клипов уже в prompts.json после обеих волн; недостающие догенерируются при повторном запуске
        log.warning("Остановка промптов: готовы промпты для %d из %d клипов, они сохранены в prompts.json.",
                    len(ready), len(needed))
    return PromptRun(ready=ready, state=state, errors=errors, abort=aborts[0] if aborts else None)
