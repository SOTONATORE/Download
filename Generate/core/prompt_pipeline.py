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
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0                  # сколько раз вызывали транспорт в этой волне
        self.last_transient = False     # последний вызов завершился временным сбоем

    def generate_json(self, model, system, user, schema):
        self.calls += 1
        try:
            text = self._inner.generate_json(model, system, user, schema)
        except gp.GeminiRuntimeError as exc:
            self.last_transient = go.is_transient(exc)
            raise
        self.last_transient = False
        return text

    def __getattr__(self, name):  # прочие атрибуты (например, active_model) берём у внутреннего транспорта
        return getattr(self._inner, name)


def _is_hard(exc, watch: "_TransportWatch") -> bool:
    """Жёсткая ошибка: любой сбой, кроме временного, и ошибка шага после успешных вызовов.

    Временный сбой (is_transient) после исчерпания повторов или без вызовов транспорта остаётся
    возвратом к сохранённым промптам, как раньше.
    """
    if not go.is_transient(exc):
        return True
    return watch.calls > 0 and not watch.last_transient


def _abort_text(wave: str, exc) -> str:
    """«волна, сегмент N» для строки об остановке; номер берётся из текста ошибки шага (Сегмент N:)."""
    m = re.search(r"Сегмент (\d+):", str(exc))
    return f"{wave}, сегмент {m.group(1) if m else 'н/д'}"


def _video_step(needed: list, *, segments, cfg, transport, prompts_path, profile, timing,
                gemini_options: dict, errors: list, aborts: list) -> dict:
    """Промпты для видео: generate_and_store дописывает prompts.json после каждого батча."""
    watch = _TransportWatch(transport)
    try:
        return go.generate_and_store(
            segments, needed, cfg, watch, prompts_path,
            profile=profile, timing=timing, **gemini_options)
    except gp.GeminiRuntimeError as exc:
        errors.append(str(exc))
        if _is_hard(exc, watch):
            aborts.append(_abort_text("видео", exc))
            log.error("Генерация промптов (видео) остановлена: %s.", exc)
            return read_state(prompts_path)
        log.error("Генерация промптов прервана: %s. Используем уже сохранённые.", exc)
        return read_state(prompts_path)


def _photo_step(needed: list, *, segments, photo_profile, photo_cfg, transport, prompts_path,
                gemini_options: dict, errors: list, aborts: list) -> None:
    """Промпты для фото: профиль photo_profile, без тайминга; num_frames всегда 1 (STEP 7, блок B2)."""
    watch = _TransportWatch(transport)
    try:
        state = go.load_state(prompts_path, segments, photo_profile)
        prompts = state["prompts"]
        pending = go.missing_segments(prompts, needed)
        if not pending:
            return
        managed = go.ManagedTransport(
            watch, gemini_options.get("models", go.DEFAULT_MODELS),
            rpm_limit=gemini_options.get("rpm_limit", 15), tpm_limit=gemini_options.get("tpm_limit", 250_000),
            rpd_limit=gemini_options.get("rpd_limit", 500))

        def on_batch(results: list) -> None:
            for r in results:
                old = prompts.get(str(r.num)) or {}
                prompts[str(r.num)] = {
                    "num": r.num, "num_frames": PHOTO_NUM_FRAMES, "prompt": r.prompt,
                    "status": "ready" if r.prompt.strip() else "failed",
                    "attempts": int(old.get("attempts", 0)) + 1,
                    "error": r.review_reason, "needs_review": r.needs_review, "extra": r.extra,
                }
            state["model"] = managed.active_model
            state["prompts"] = {k: prompts[k] for k in sorted(prompts, key=int)}
            go.atomic_write_json(prompts_path, state)

        # провайдер у транспорта (через _TransportWatch); для gemini аргумент не передаётся, вызов как раньше
        provider = getattr(watch, "provider", "gemini")
        extra = {} if provider == "gemini" else {"provider": provider}
        gp.generate_prompts(segments, pending, photo_cfg, managed, managed.active_model, on_batch=on_batch, **extra)
    except gp.GeminiRuntimeError as exc:
        errors.append(str(exc))
        if _is_hard(exc, watch):
            aborts.append(_abort_text("фото", exc))
            log.error("Генерация промптов для фото остановлена: %s.", exc)
            return
        log.error("Генерация промптов для фото прервана: %s. Используем уже сохранённые.", exc)


def build_prompts(needed: list, *, segments: list, photo_nums, cfg, profile: str, timing,
                  photo_profile: Optional[str], photo_cfg, transport, gemini_options: dict,
                  prompts_path) -> PromptRun:
    """Единый конвейер промптов для списка номеров needed.

    photo_profile None: все номера идут в видео (photo_nums не читается).
    Иначе номера из photo_nums идут в фото, остальные в видео; сначала видео, потом фото,
    затем состояние читается из файла (оно содержит оба вида).
    Порядок и аргументы вызовов совпадают с прежним Orchestrator._generate.
    """
    errors: list = []
    aborts: list = []  # жёсткие ошибки волн; решение об остановке принимает оркестратор после обеих волн
    if photo_profile is None:
        state = _video_step(needed, segments=segments, cfg=cfg, transport=transport,
                            prompts_path=prompts_path, profile=profile, timing=timing,
                            gemini_options=gemini_options, errors=errors, aborts=aborts)
    else:
        video = [n for n in needed if n not in photo_nums]
        photo = [n for n in needed if n in photo_nums]
        if video:
            _video_step(video, segments=segments, cfg=cfg, transport=transport,
                        prompts_path=prompts_path, profile=profile, timing=timing,
                        gemini_options=gemini_options, errors=errors, aborts=aborts)
        if photo:
            _photo_step(photo, segments=segments, photo_profile=photo_profile, photo_cfg=photo_cfg,
                        transport=transport, prompts_path=prompts_path,
                        gemini_options=gemini_options, errors=errors, aborts=aborts)
        state = read_state(prompts_path)  # после фото: состояние файла с обоими видами
    prompts = state.setdefault("prompts", {})
    ready = [n for n in needed if _prompt_ready(prompts.get(str(n)))]
    return PromptRun(ready=ready, state=state, errors=errors, abort=aborts[0] if aborts else None)
