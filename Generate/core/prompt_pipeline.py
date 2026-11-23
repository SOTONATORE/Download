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
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Optional

try:  # пакетный и «плоский» импорт (как в остальных модулях core)
    from . import gemini_orchestrator as go
    from . import gemini_prompts as gp
    from . import config_loader as cl
    from . import duplicate_check as dc
    from . import prompt_check as pc
    from . import segment_split as ss
    from . import video_value as vv
    from .timing import compute_timings
except ImportError:  # pragma: no cover
    import gemini_orchestrator as go  # type: ignore
    import gemini_prompts as gp  # type: ignore
    import config_loader as cl  # type: ignore
    import duplicate_check as dc  # type: ignore
    import prompt_check as pc  # type: ignore
    import segment_split as ss  # type: ignore
    import video_value as vv  # type: ignore
    from timing import compute_timings  # type: ignore

# тот же логгер, что у оркестратора: строки журнала не меняются
log = logging.getLogger("Generate.orchestrator")

PHOTO_NUM_FRAMES = 1        # фото: один промпт = один кадр; задаётся кодом, не из тайминга
VIDEO_WAVE = "видео"        # имя волны видео в общем пуле батчей
PHOTO_WAVE = "фото"         # имя волны фото в общем пуле батчей
DUP_REWORK_MAX_PASSES = 3   # сколько проходов дедупликации делается, если переписанный клип совпал с другим
# cartoon_photos (batch_passage): монотонность соседних клипов — окно из стольких номеров и порог сходства сцены
MONOTONY_WINDOW = 3
MONOTONY_JACCARD = 0.85
# Русский: повторяемое описание героя — не сцена; «the main character» — роль, а не сцена (глаголы и предметы остаются)
_APPEARANCE_RX = re.compile(r"the main character has[^.]*\.|\bthe main character\b", re.I)
# Русский: типовой текст персонажей, наряда и стиля (style.md повторяет его в каждом промпте) не должен занижать
# различие сцен. Фразы ищутся независимо от порядка клауз и от «with/and» перед ними; предметы, места и действия
# (stakes, twine, digging fork, basket, cellar, farm shop и т. п.) сюда не входят.
_BOILERPLATE_RX = re.compile(
    # старые правила «до точки»: предложение целиком типовое, сохранены без изменений
    r"has warm natural skin tones[^.]*\.|has proportional cartoon limbs[^.]*\.|small black dot eyes[^.]*\.|"
    # голова и тело главного героя
    r"(?:(?:with|and)\s+)?a\s+large\s+smooth\s+pure-white\s+round\s+head|"
    r"(?:(?:with|and)\s+)?a\s+crisp\s+medium-fine\s+black\s+ink\s+outline|"
    r"(?:(?:with|and)\s+)?small\s+(?:solid\s+)?black\s+dot\s+eyes|"
    r"(?:(?:and|with)\s+)?a\s+plain(?:\s+thin)?\s+solid-black\s+stick\s+(?:torso|body)|"
    r"(?:(?:with|and)\s+)?(?:thin\s+|simple\s+)?black\s+stick\s+(?:arms|hands|legs)|"
    # мимика: один рот на лице
    r"(?:a\s+)?(?:closed-line\s+(?:smile|mouth)|(?:worried\s+wavy|wavy\s+worried|open\s+O-shaped)\s+mouth)|"
    r"one\s+mouth\s+expression(?:\s+showing)?|"
    # рост в кадре (доля высоты кадра)
    r"(?:at\s+about\s+)?(?:\d+|twenty|forty|fifty|sixty)\s+percent\s+of\s+image\s+height|"
    # взрослые персонажи и макро-руки: пропорции, кожа, волосы, нос, брови
    r"(?:a\s+)?head\s+about\s+one\s+fifth\s+of\s+body\s+height|"
    r"(?:a\s+)?proportional\s+(?:cartoon|clothed|2d)\s+(?:body|figure|limbs)|"
    r"(?:(?:with|and)\s+)?warm\s+natural\s+skin\s+tones|"
    r"(?:(?:with|and)\s+)?simple\s+rounded\s+fingers|"
    r"simple\s+(?:solid(?:-colou?r)?\s+|(?:brown|dark|black|light|gray|grey)\s+)?hair|"
    r"(?:a\s+)?small\s+curved-line\s+nose|"
    r"(?:(?:with|and)\s+)?(?:raised|relaxed|neutral|expressive|lifted)(?:\s+expressive)?(?:\s+thin\s+curved)?\s+eyebrows|"
    # контур и заливка из общего стиля
    r"(?:smooth|crisp)\s+closed\s+(?:black\s+)?(?:ink\s+)?outlines|"
    r"flat\s+(?:matte\s+)?(?:earthy\s+|muted\s+)?fills|"
    # фиксированные наряды ролей (одежда одна на весь ролик)
    r"(?:(?:wearing|wears)\s+)?(?:"
    r"muted\s+rust\s+plaid\s+flannel\s+shirt,?\s+medium-blue\s+denim\s+overalls\s+and\s+brown\s+boots|"
    r"muted\s+sage\s+sweater\s+and\s+medium-brown\s+trousers|"
    r"white\s+lab\s+coat\s+and\s+eyeglasses|"
    r"white\s+jacket\s+and\s+tall\s+white\s+cap|"
    r"muted\s+brown\s+work\s+shirt,?\s+tan\s+trousers,?\s+brown\s+boots\s+and\s+a\s+tan\s+hard\s+hat|"
    r"period\s+tunic\s+and\s+trousers)",
    re.I)
# Русский: серия из трёх подряд клипов одного вида кадра (setting до «:») с перекрытием слов соседей
MONOTONY_RUN = 3
MONOTONY_RUN_JACCARD = 0.65
_WORD_RX = re.compile(r"[a-z0-9']+")
# Русский: анти-клон сцены (batch_passage) — сравнение с последними CLONE_WINDOW клипами того же вида кадра
CLONE_WINDOW = 100
CLONE_JACCARD = 0.65
# Русский: общие слова стиля и поверхности — не признак сцены; предметы, здания, инструменты и культуры остаются
_FILLER_WORDS = frozenset({
    "warm", "cream", "pale", "flat", "matte", "crisp", "closed", "outline", "outlines", "fills", "cel",
    "shading", "clean", "2d", "single", "layer", "soft", "shadow", "strip", "underside", "backdrop", "surface",
})
_HAND_DRAWN_RX = re.compile(r"hand-drawn", re.IGNORECASE)


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
    # ключ openrouter_sends_per_window не принимает generate_and_store через **go_options: он передаётся отдельно
    go_options = {k: v for k, v in gemini_options.items() if k != "openrouter_sends_per_window"}
    sends = gemini_options.get("openrouter_sends_per_window")
    sends_kw = {} if sends is None else {"sends_per_window": sends}
    try:
        return go.generate_and_store(
            segments, needed, cfg, watch, prompts_path,
            profile=profile, timing=timing, managed=managed, state=state, lock=lock, **pool_kw, **go_options,
            **sends_kw)
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
        # очередь исправлений фото из сохранённого состояния (под замком: словарь пишут обе волны)
        with guard:
            seeds = go._rework_seeds(prompts, photo=True, only=set(needed), **go.rework_kwargs(photo_cfg))
        if not pending and not seeds:
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
                    if getattr(r, "rework_pending", None):
                        # клип в очереди исправлений: сырой текст отдельно; в поле prompt — со стилем
                        entry["prompt_raw"] = r.prompt_raw
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
        # очередь прошлого запуска передаётся только когда она не пуста: вызов без неё остаётся прежним
        rework_kw = {} if not seeds else {"rework": seeds}
        # число отправок OpenRouter из gemini_options; для gemini ключа нет, аргумент не передаётся
        sends = gemini_options.get("openrouter_sends_per_window")
        sends_kw = {} if sends is None else {"sends_per_window": sends}
        gp.generate_prompts(segments, pending, photo_cfg, managed, managed.active_model, on_batch=on_batch,
                            **extra, **pool_kw, **rework_kw, **sends_kw)
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


def _pool_threads(provider: str, sends_per_window: Optional[int] = None) -> int:
    """Всего потоков общего пула: число ключей провайдера, не меньше 1 и не больше MAX_PARALLEL_BATCHES."""
    try:
        keys = gp._provider_key_count(provider)
    except Exception:  # ключей нет или не прочитались: как в generate_prompts, один поток
        keys = 0
    if provider == "openrouter":
        # OpenRouter: ключи × openrouter_sends_per_window из конфига, без общего ограничения MAX_PARALLEL_BATCHES
        return gp.openrouter_thread_count(keys, sends_per_window)
    return min(max(keys, 1), gp.MAX_PARALLEL_BATCHES)


def _photo_only_pool(watch, gemini_options: dict):
    """Общий пул для прогона, где есть только фото-волна (видео нет).

    Потоки те же, что фото получает в двух волнах: OpenRouter — ключи × openrouter_sends_per_window,
    прочие провайдеры — как _pool_threads. per_wave_cap ограничивает их числом батчей волны прямо в
    arrive_and_wait, поэтому число батчей здесь заранее не нужно. Без пула исправления очереди начинались
    бы только после конца всех главных батчей; пул даёт им выполняться раньше.
    """
    provider = getattr(watch, "provider", "gemini")
    sends = gemini_options.get("openrouter_sends_per_window")
    if provider == "openrouter":
        try:
            keys = gp._provider_key_count(provider)
        except Exception:  # ключей нет или не прочитались: как в _run_waves_together, ноль
            keys = 0
        threads = gp.openrouter_thread_count(keys, sends)
    else:
        threads = _pool_threads(provider, sends)
    return gp.SharedBatchPool({PHOTO_WAVE: threads}, bind=watch.bind, per_wave_cap=True)


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
    sends = gemini_options.get("openrouter_sends_per_window")
    if provider == "openrouter":
        # OpenRouter: каждой волне — ключи × отправки; потоки волны не больше её батчей (per_wave_cap в пуле).
        # Лимит отправок по каждому ключу держит транспорт, поэтому лишние потоки только ждут.
        try:
            keys = gp._provider_key_count(provider)
        except Exception:  # ключей нет или не прочитались: как в generate_prompts, ноль
            keys = 0
        per_wave = gp.openrouter_thread_count(keys, sends)
        log.info("Потоки OpenRouter: ключей %d, отправок в окне %s, потоков на волну до %d (не больше батчей волны).",
                 keys, sends if sends is not None else "не задано", per_wave)
        pool = gp.SharedBatchPool({VIDEO_WAVE: per_wave, PHOTO_WAVE: per_wave}, bind=watch.bind, per_wave_cap=True)
    else:
        total = _pool_threads(provider, sends)
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


def _raw_prompt(styled: str, style_text: str) -> str:
    """Сырой текст клипа без стиля: обратная операция attach_style (стиль дописан через «. »)."""
    text = (styled or "").strip()
    if style_text and text.endswith(style_text):
        text = text[: -len(style_text)].rstrip()
        if text.endswith("."):
            text = text[:-1].rstrip()
    return text


def _with_dup_niche(cfg, dup_pass: int):
    """Отдельный суффикс niche на время прохода дедупликации, чтобы дампы не перезаписывали основной раунд."""
    if cfg is None or not getattr(cfg, "niche", ""):
        return cfg
    if is_dataclass(cfg):
        return replace(cfg, niche=f"{cfg.niche}_dup{dup_pass}")
    return cfg


def mark_duplicates_for_rework(prompts: dict, nums, style_text: str, *,
                               allow_extra_round: bool = False) -> list:
    """Клипы из групп одинаковых промптов, кроме первого номера группы, ставятся в очередь исправлений.

    Меняет записи prompts (словарь состояния; вызывать под замком). Первый номер группы (самый малый) не трогается.
    Клип с непустой очередью не меняется: его нарушение уже исправляется, повтор дубликата проверит барьер.
    При allow_extra_round=False (по умолчанию) клип на пределе кругов (REPAIR_ROUNDS) не ставится; при
    allow_extra_round=True (цикл _duplicate_rework) ему выделяется чистый счётчик кругов, чтобы он не остался
    дубликатом после прошлых исправлений содержания. Возвращает номера, поставленные в очередь.
    """
    texts = {str(n): str((prompts.get(str(n)) or {}).get("prompt", "")) for n in sorted(set(nums))}
    queued: list = []
    for group in dc.find_duplicate_prompts(texts):
        first = int(group[0])
        for key in group[1:]:
            entry = prompts.get(key)
            if not isinstance(entry, dict) or entry.get("rework_pending"):
                continue
            rework_round = int(entry.get("rework_round", 0) or 0)
            if rework_round >= gp.REPAIR_ROUNDS and not allow_extra_round:
                continue
            if allow_extra_round:
                rework_round = 0
            entry["prompt_raw"] = _raw_prompt(str(entry.get("prompt", "")), style_text)
            entry["rework_pending"] = [pc.make_problem(pc.PROBLEM_DUPLICATE_PROMPT, str(first))]
            entry["rework_round"] = rework_round
            queued.append(int(key))
    return queued


def _clip_setting(entry: dict) -> str:
    """Setting клипа batch_passage в нижнем регистре, без лишних пробелов; пусто, если его нет."""
    extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
    return " ".join(str(extra.get("setting") or "").split()).lower()


def _scene_words(entry: dict, style_text: str) -> frozenset:
    """Слова сцены клипа без стиля, без повторяемой фразы о внешности героя и без типового текста персонажей."""
    raw = _raw_prompt(str(entry.get("prompt", "")), style_text)
    cleaned = _BOILERPLATE_RX.sub(" ", _APPEARANCE_RX.sub(" ", raw))
    cleaned = _HAND_DRAWN_RX.sub(" ", cleaned)   # «hand-drawn» — один стилевой термин, а не два слова
    return frozenset(_WORD_RX.findall(cleaned.lower())) - _FILLER_WORDS


def _visual_mode(entry: dict) -> str:
    """Вид кадра клипа (часть setting до «:», например macro, medium); пусто, если его нет."""
    setting = _clip_setting(entry)
    return setting.split(":", 1)[0].strip() if ":" in setting else ""


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Доля общих слов; пусто — 0.0."""
    return len(a & b) / len(a | b) if a and b else 0.0


def mark_monotony_for_rework(prompts: dict, nums, style_text: str) -> list:
    """cartoon_photos (batch_passage): соседние клипы монотонны — ставятся в очередь исправлений.

    Окно: клип сравнивается с двумя предыдущими номерами волны (в числовом порядке). Монотонность — одинаковый
    непустой setting или сцена с долей общих слов не ниже MONOTONY_JACCARD. Клип с очередью или на пределе кругов
    (REPAIR_ROUNDS) пропускается, поэтому цикл не бесконечен. Текст повтора — DUPLICATE_REWORK_TEXT (через код
    дубликата, без новых кодов).

    Русский: после окна и серии — анти-клон между пакетами: клип с тем же видом кадра (setting до «:») и сценой
    с долей общих слов не ниже CLONE_JACCARD среди последних CLONE_WINDOW клипов (уже принятых, без очереди)
    ставится в очередь с маркером «mode=<вид>»; _rework_text даёт для него свою английскую подсказку.
    Соседи из локального окна MONOTONY_WINDOW в анти-клон не входят: для них действуют прежние правила.
    Меняет prompts под замком вызывающего. Возвращает номера, поставленные в очередь.
    """
    ordered = sorted(set(int(n) for n in nums))
    queued: list = []
    for idx, n in enumerate(ordered):
        entry = prompts.get(str(n))
        if not isinstance(entry, dict) or entry.get("rework_pending"):
            continue
        if int(entry.get("rework_round", 0) or 0) >= gp.REPAIR_ROUNDS:
            continue
        setting = _clip_setting(entry)
        mode = _visual_mode(entry)
        words = _scene_words(entry, style_text)
        hit = None
        for earlier in ordered[max(0, idx - (MONOTONY_WINDOW - 1)):idx]:
            other = prompts.get(str(earlier)) or {}
            same_setting = bool(setting) and setting == _clip_setting(other)
            other_words = _scene_words(other, style_text)
            similar = bool(words and other_words) and len(words & other_words) / len(words | other_words) >= MONOTONY_JACCARD
            if same_setting or similar:
                hit = earlier
                break
        if hit is None and idx >= MONOTONY_RUN - 1:
            # Русский: третий и далее подряд клип того же вида кадра с перекрытием соседей — тоже монотонность
            run = ordered[idx - (MONOTONY_RUN - 1):idx + 1]
            run_entries = [prompts.get(str(x)) or {} for x in run]
            modes = {_visual_mode(e) for e in run_entries}
            run_words = [_scene_words(e, style_text) for e in run_entries]
            pairs_ok = all(_jaccard(run_words[k], run_words[k + 1]) >= MONOTONY_RUN_JACCARD
                           for k in range(len(run_words) - 1))
            if len(modes) == 1 and "" not in modes and pairs_ok:
                hit = run[-2]
        marker = ""
        if hit is None and mode and words:
            # Русский: анти-клон — тот же вид кадра и почти та же сцена в последних CLONE_WINDOW клипах, кроме соседей
            # из локального окна (их проверяют правила выше, прежний порог); поэтому срез до idx - (MONOTONY_WINDOW - 1)
            for earlier in ordered[max(0, idx - CLONE_WINDOW):max(0, idx - (MONOTONY_WINDOW - 1))]:
                other = prompts.get(str(earlier))
                if not isinstance(other, dict) or other.get("rework_pending") or _visual_mode(other) != mode:
                    continue
                if _jaccard(words, _scene_words(other, style_text)) >= CLONE_JACCARD:
                    hit, marker = earlier, mode
                    break
        if hit is not None:
            fragment = f"{hit} mode={marker}" if marker else str(hit)
            entry["prompt_raw"] = _raw_prompt(str(entry.get("prompt", "")), style_text)
            entry["rework_pending"] = [pc.make_problem(pc.PROBLEM_DUPLICATE_PROMPT, fragment)]
            queued.append(n)
    return queued


def _duplicate_rework(needed: list, *, photo_nums, segments, cfg, photo_cfg, profile: str,
                      photo_profile: Optional[str], timing, watch, prompts_path, gemini_options: dict,
                      errors: list, aborts: list, state, lock) -> dict:
    """Ниши с ключом fail_on_duplicate_prompts: одинаковые промпты разных клипов идут в очередь исправлений.

    Выполняется после обеих волн генерации и до итогового состояния. Для профилей без ключа ничего не делает
    (ни чтения состояния, ни вызова модели). Исправление повторяется до DUP_REWORK_MAX_PASSES проходов, чтобы
    переписанный клип, совпавший с другим клипом прогона, тоже был переписан. Итоговую проверку дубликатов
    делает verify_prompts после этого этапа.
    """
    if state is None:
        state = read_state(prompts_path)
    photo = [n for n in needed if n in photo_nums] if photo_profile is not None else []
    photo_set = set(photo)
    video = [n for n in needed if n not in photo_set]
    dup_photo = bool(photo) and cl.load_profile_fail_on_duplicate_prompts(photo_profile)
    dup_video = bool(video) and cl.load_profile_fail_on_duplicate_prompts(profile)
    if not (dup_photo or dup_video):
        return state
    style_video = gp.preset_text(cfg.preset, cfg.niche_preset_file) if dup_video else ""
    style_photo = gp.preset_text(photo_cfg.preset, photo_cfg.niche_preset_file) if dup_photo else ""
    managed = None
    for dup_pass in range(1, DUP_REWORK_MAX_PASSES + 1):
        with lock:
            prompts = state.setdefault("prompts", {})
            queued_video = mark_duplicates_for_rework(
                prompts, video, style_video, allow_extra_round=True) if dup_video else []
            queued_photo = mark_duplicates_for_rework(
                prompts, photo, style_photo, allow_extra_round=True) if dup_photo else []
            # Русский: монотонность только для batch_passage; точные дубликаты уже поставлены выше и пропускаются
            if dup_video and getattr(cfg, "batch_passage", False):
                queued_video = sorted(set(queued_video) | set(mark_monotony_for_rework(prompts, video, style_video)))
            if dup_photo and getattr(photo_cfg, "batch_passage", False):
                queued_photo = sorted(set(queued_photo) | set(mark_monotony_for_rework(prompts, photo, style_photo)))
            if not queued_video and not queued_photo:
                return state
            state["prompts"] = {k: prompts[k] for k in sorted(prompts, key=int)}
            go.atomic_write_json(prompts_path, state)
        queued = sorted(queued_video + queued_photo)
        log.info("Дубликаты промптов (проход %d из %d): в очередь исправлений поставлено клипов %d: %s.",
                 dup_pass, DUP_REWORK_MAX_PASSES, len(queued), gp._num_ranges(queued), extra=gp.KEY_EVENT_EXTRA)
        if aborts:
            loaded = read_state(prompts_path)
            return loaded if loaded.get("prompts") else state
        if managed is None:
            managed = go.ManagedTransport(
                watch, gemini_options.get("models", go.DEFAULT_MODELS),
                rpm_limit=gemini_options.get("rpm_limit", 15), tpm_limit=gemini_options.get("tpm_limit", 250_000),
                rpd_limit=gemini_options.get("rpd_limit", 500))
        dup_cfg = _with_dup_niche(cfg, dup_pass)
        dup_photo_cfg = _with_dup_niche(photo_cfg, dup_pass)
        if queued_video:
            state = _video_step(queued_video, segments=segments, cfg=dup_cfg, watch=watch, prompts_path=prompts_path,
                                profile=profile, timing=timing, gemini_options=gemini_options, errors=errors,
                                aborts=aborts, managed=managed, state=state, lock=lock)
        if queued_photo and not aborts:
            _photo_step(queued_photo, segments=segments, photo_cfg=dup_photo_cfg, watch=watch, prompts_path=prompts_path,
                        gemini_options=gemini_options, errors=errors, aborts=aborts, managed=managed,
                        state=state, lock=lock)
        loaded = read_state(prompts_path)
        if loaded.get("prompts"):
            state = loaded
        if aborts:
            return state
    with lock:
        prompts = state.get("prompts", {})
        leftover = []
        if dup_video:
            leftover.extend(dc.find_duplicate_prompts(
                {str(n): str((prompts.get(str(n)) or {}).get("prompt", "")) for n in sorted(set(video))}))
        if dup_photo:
            leftover.extend(dc.find_duplicate_prompts(
                {str(n): str((prompts.get(str(n)) or {}).get("prompt", "")) for n in sorted(set(photo))}))
    if leftover:
        groups_text = "; ".join(", ".join(str(x) for x in g) for g in leftover)
        log.warning("Дубликаты промптов: после %d проходов остались совпадающие группы: %s.",
                    DUP_REWORK_MAX_PASSES, groups_text)
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
    # Русский: клипы «на проверку» из прошлого запуска пересматриваются по текущим правилам до решения,
    # какие волны идти (run_video / run_photo); вызова модели нет. Фото — только при photo_profile.
    go.recover_resolved_file(prompts_path, segments, cfg, photo_cfg if photo_profile is not None else None)
    if photo_profile is None:
        state = _video_step(list(needed), segments=segments, cfg=cfg, watch=watch,
                            prompts_path=prompts_path, profile=profile, timing=timing,
                            gemini_options=gemini_options, errors=errors, aborts=aborts,
                            managed=None, state=None, lock=lock)
    else:
        video = [n for n in needed if n not in photo_nums]
        photo = [n for n in needed if n in photo_nums]
        # очередь исправлений прошлого запуска: волна с очередью идёт, даже если клипов для генерации у неё нет
        # (только чтение файла, без замка и без вызовов модели: потоков волн ещё нет)
        prev_prompts = read_state(prompts_path).get("prompts", {})
        video_seeds = bool(go._rework_seeds(prev_prompts, only=set(video), **go.rework_kwargs(cfg)))
        photo_seeds = bool(go._rework_seeds(prev_prompts, photo=True, only=set(photo),
                                            **go.rework_kwargs(photo_cfg)))
        run_video = bool(video) or video_seeds
        run_photo = bool(photo) or photo_seeds
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
            if run_video:
                state = _load_shared(state, prompts_path, segments, file_profile)
                state = _video_step(video, segments=segments, cfg=cfg, watch=watch,
                                    prompts_path=prompts_path, profile=profile, timing=timing,
                                    gemini_options=gemini_options, errors=errors, aborts=aborts,
                                    managed=managed, state=state, lock=lock)
            if run_photo:
                state = _load_shared(state, prompts_path, segments, file_profile)
                # Русский: фото без видео тоже через общий пул, чтобы очередь исправлений шла до конца главных батчей
                _photo_step(photo, segments=segments, photo_cfg=photo_cfg, watch=watch,
                            prompts_path=prompts_path, gemini_options=gemini_options,
                            errors=errors, aborts=aborts, managed=managed, state=state, lock=lock,
                            pool=_photo_only_pool(watch, gemini_options))
        state = read_state(prompts_path)  # после обеих волн: состояние файла с обоими видами
    # Русский: дубликаты промптов — только ниши с fail_on_duplicate_prompts; остальные профили этот этап не трогает
    state = _duplicate_rework(needed, photo_nums=photo_nums, segments=segments, cfg=cfg, photo_cfg=photo_cfg,
                              profile=profile, photo_profile=photo_profile, timing=timing, watch=watch,
                              prompts_path=prompts_path, gemini_options=gemini_options, errors=errors,
                              aborts=aborts, state=state, lock=lock)
    prompts = state.setdefault("prompts", {})
    ready = [n for n in needed if _prompt_ready(prompts.get(str(n)))]
    if aborts:
        # итог до остановки: сколько клипов уже в prompts.json после обеих волн; недостающие догенерируются при повторном запуске
        log.warning("Остановка промптов: готовы промпты для %d из %d клипов, они сохранены в prompts.json.",
                    len(ready), len(needed))
    return PromptRun(ready=ready, state=state, errors=errors, abort=aborts[0] if aborts else None)


_I2V_VALUE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "video_value": {"type": "integer"},
        },
        "required": ["id", "video_value"],
    },
}


def _video_prompt_ready(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    num_frames = entry.get("video_num_frames")
    return (entry.get("video_status") == "ready"
            and bool(str(entry.get("video_prompt", "")).strip())
            and isinstance(num_frames, int) and not isinstance(num_frames, bool)
            and num_frames > 0)


def _default_i2v_value_sender(transport, gemini_options: dict):
    managed = go.ManagedTransport(
        transport, gemini_options.get("models", go.DEFAULT_MODELS),
        rpm_limit=gemini_options.get("rpm_limit", 15),
        tpm_limit=gemini_options.get("tpm_limit", 250_000),
        rpd_limit=gemini_options.get("rpd_limit", 500))
    return lambda prompt: managed.generate_json("", "", prompt, _I2V_VALUE_SCHEMA, reasoning_effort="none")


def build_i2v_prompts(
    needed: list,
    *,
    segments: list,
    photo_cfg,
    photo_profile: str,
    video_cfg,
    video_profile: str,
    timing,
    transport,
    gemini_options: dict,
    prompts_path,
    distribution_path=None,
    intro_seconds: float = 180.0,
    intro_video_percent: float = 70.0,
    after_intro_video_percent: float = 10.0,
    min_anim_sec: float = 2.0,
    score_send=None,
    score_workers: int = 1,
) -> PromptRun:
    """Двухэтапный конвейер промптов для I2V (ниша картунов):
    1. Фото-промпты для всех needed (с дедупликацией и антимонотонностью).
    2. Оценка video_value по фото-промпту + SRT и равномерный отбор select_i2v_segments
       (при resume сохранённый выбор i2v в prompts.json не пересчитывается).
    3. Генерация I2V видео-промптов батчами только для выбранных сегментов с расчётом video_num_frames (8k+1).
    4. Раздельное хранение полей фото и видео в prompts.json.
    """
    needed_list = list(needed)
    # Этап 1: фото-промпты для ВСЕХ сегментов
    photo_run = build_prompts(
        needed_list,
        segments=segments,
        photo_nums=set(needed_list),
        cfg=photo_cfg,
        profile=photo_profile,
        timing=None,
        photo_profile=photo_profile,
        photo_cfg=photo_cfg,
        transport=transport,
        gemini_options=gemini_options,
        prompts_path=prompts_path,
    )
    if photo_run.abort:
        return photo_run

    state = photo_run.state
    prompts = state.setdefault("prompts", {})
    ready_photos = [n for n in needed_list if _prompt_ready(prompts.get(str(n)))]
    if not ready_photos:
        return photo_run

    style_photo = gp.preset_text(photo_cfg.preset, photo_cfg.niche_preset_file)
    by_num = {s.num: s for s in segments}

    # Этап 2: выбор I2V-сегментов (при resume сохранённый выбор i2v не пересчитывается)
    has_saved_selection = bool(needed_list) and all(
        isinstance((prompts.get(str(n)) or {}).get("i2v"), bool) for n in needed_list
    )
    if has_saved_selection:
        i2v_nums = [n for n in needed_list if (prompts.get(str(n)) or {}).get("i2v") is True]
        log.info("I2V-отбор (resume): сохранён выбор из prompts.json — видео %d из %d сегментов.",
                 len(i2v_nums), len(needed_list))
    else:
        ss_segs = ss.to_segments(segments, needed_list)
        raw_photo_map = {
            n: _raw_prompt(str((prompts.get(str(n)) or {}).get("prompt", "")), style_photo)
            for n in needed_list
        }
        eligible_segs = [s for s in ss_segs if s.duration >= min_anim_sec - 1e-9]
        if not eligible_segs or (intro_video_percent == 0 and after_intro_video_percent == 0):
            values = {s.id: 0 for s in eligible_segs}
        else:
            send_fn = score_send if score_send is not None else _default_i2v_value_sender(transport, gemini_options)
            try:
                values = vv.score_i2v_segments(
                    ss_segs, raw_photo_map, send_fn, workers=score_workers, min_anim_sec=min_anim_sec,
                )
            except ValueError as exc:
                raise gp.GeminiRuntimeError(
                    f"Оценка I2V video_value не удалась: {exc} Запуск остановлен, запасного значения нет."
                ) from None
        dist = ss.select_i2v_segments(
            ss_segs, values,
            intro_seconds=intro_seconds,
            intro_video_percent=intro_video_percent,
            after_intro_video_percent=after_intro_video_percent,
            min_anim_sec=min_anim_sec,
        )
        if distribution_path is not None:
            vv.write_distribution(distribution_path, dist, ss_segs, values)
            log.info("I2V-распределение записано в файл: %s.", distribution_path)
        i2v_nums = list(dist["video"])
        i2v_set = set(i2v_nums)
        for n in needed_list:
            entry = prompts.get(str(n))
            if isinstance(entry, dict):
                entry["i2v"] = n in i2v_set
        state["i2v_mode"] = True
        state["video_profile"] = video_profile
        state["prompts"] = {k: prompts[k] for k in sorted(prompts, key=int)}
        go.atomic_write_json(prompts_path, state)
        log.info("I2V-отбор готов: видео %d из %d сегментов (вступление %.0f с: %.0f%%, далее %.0f%%, порог %.1f с).",
                 len(i2v_nums), len(needed_list), intro_seconds, intro_video_percent,
                 after_intro_video_percent, min_anim_sec)

    # Этап 3: генерация I2V видео-промптов только для выбранных сегментов
    pending_i2v = [n for n in i2v_nums if n in set(ready_photos) and not _video_prompt_ready(prompts.get(str(n)))]
    errors = list(photo_run.errors)
    aborts: list = []

    if pending_i2v:
        timings = {
            t.num: t for t in compute_timings(
                segments, timing.fps, timing.frame_rule,
                timing.clip_min_sec, timing.clip_max_sec,
                timing.gpu_max_sec, timing.tail_pad_sec,
            )
        }
        source_photo_prompts = {
            n: _raw_prompt(str((prompts.get(str(n)) or {}).get("prompt", "")), style_photo)
            for n in pending_i2v
        }
        segment_durations = {
            n: (by_num[n].end_ms - by_num[n].start_ms) / 1000.0
            for n in pending_i2v if n in by_num
        }
        i2v_cfg = replace(
            video_cfg,
            source_photo_prompts=source_photo_prompts,
            segment_durations=segment_durations,
        )
        watch = _TransportWatch(transport)
        lock = threading.Lock()
        managed = go.ManagedTransport(
            watch, gemini_options.get("models", go.DEFAULT_MODELS),
            rpm_limit=gemini_options.get("rpm_limit", 15),
            tpm_limit=gemini_options.get("tpm_limit", 250_000),
            rpd_limit=gemini_options.get("rpd_limit", 500),
        )
        mark = watch.calls

        def on_video_batch(results: list) -> None:
            with lock:
                for r in results:
                    old = dict(prompts.get(str(r.num)) or {})
                    old["i2v"] = True
                    old["video_prompt"] = r.prompt
                    old["video_num_frames"] = timings[r.num].num_frames if r.num in timings else 1
                    old["video_status"] = "ready" if r.prompt.strip() else "failed"
                    old["video_needs_review"] = r.needs_review
                    old["video_error"] = r.review_reason
                    old["video_attempts"] = int(old.get("video_attempts", 0)) + 1
                    prompts[str(r.num)] = old
                state["model"] = managed.active_model
                state["i2v_mode"] = True
                state["video_profile"] = video_profile
                state["prompts"] = {k: prompts[k] for k in sorted(prompts, key=int)}
                go.atomic_write_json(prompts_path, state)

        provider = getattr(watch, "provider", "gemini")
        extra = {} if provider == "gemini" else {"provider": provider}
        sends = gemini_options.get("openrouter_sends_per_window")
        sends_kw = {} if sends is None else {"sends_per_window": sends}
        try:
            gp.generate_prompts(
                segments, pending_i2v, i2v_cfg, managed, managed.active_model,
                on_batch=on_video_batch, **extra, **sends_kw,
            )
        except gp.GeminiRuntimeError as exc:
            errors.append(str(exc))
            if _is_hard(exc, watch, mark):
                aborts.append(_abort_text("I2V-видео", exc))
                log.error("Генерация I2V-промптов остановлена: %s.", exc)
            else:
                log.error("Генерация I2V-промптов прервана: %s. Используем уже сохранённые.", exc)
        loaded = read_state(prompts_path)
        if loaded.get("prompts"):
            state = loaded

    prompts = state.setdefault("prompts", {})
    ready = [n for n in needed_list if _prompt_ready(prompts.get(str(n)))]
    return PromptRun(ready=ready, state=state, errors=errors, abort=aborts[0] if aborts else None)