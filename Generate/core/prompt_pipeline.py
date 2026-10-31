"""Общий конвейер промптов для видео и фото (STEP 7, блок B2; вынесено из orchestrator.py).

Единственное место, где готовятся промпты. Генерация (Orchestrator._generate) и проверка
(cli.prepare_prompts_only, workflow verify_gemini) вызывают build_prompts с одними и теми же входами.
Проверки запрещённых типов кадров и незаполненных маркеров, а также переписывание (rework loop)
выполняются внутри gemini_prompts.generate_prompts (шаг 6), поэтому идут по этому пути автоматически.
"""
from __future__ import annotations

import json
import logging
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
    errors: тексты остановок шагов Gemini (GeminiRuntimeError), которые были перехвачены и записаны в журнал.
    """
    ready: list
    state: dict
    errors: list = field(default_factory=list)


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


def _video_step(needed: list, *, segments, cfg, transport, prompts_path, profile, timing,
                gemini_options: dict, errors: list) -> dict:
    """Промпты для видео: generate_and_store дописывает prompts.json после каждого батча."""
    try:
        return go.generate_and_store(
            segments, needed, cfg, transport, prompts_path,
            profile=profile, timing=timing, **gemini_options)
    except gp.GeminiRuntimeError as exc:
        log.error("Генерация промптов прервана: %s. Используем уже сохранённые.", exc)
        errors.append(str(exc))
        return read_state(prompts_path)


def _photo_step(needed: list, *, segments, photo_profile, photo_cfg, transport, prompts_path,
                gemini_options: dict, errors: list) -> None:
    """Промпты для фото: профиль photo_profile, без тайминга; num_frames всегда 1 (STEP 7, блок B2)."""
    try:
        state = go.load_state(prompts_path, segments, photo_profile)
        prompts = state["prompts"]
        pending = go.missing_segments(prompts, needed)
        if not pending:
            return
        managed = go.ManagedTransport(
            transport, gemini_options.get("models", go.DEFAULT_MODELS),
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

        gp.generate_prompts(segments, pending, photo_cfg, managed, managed.active_model, on_batch=on_batch)
    except gp.GeminiRuntimeError as exc:
        log.error("Генерация промптов для фото прервана: %s. Используем уже сохранённые.", exc)
        errors.append(str(exc))


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
    if photo_profile is None:
        state = _video_step(needed, segments=segments, cfg=cfg, transport=transport,
                            prompts_path=prompts_path, profile=profile, timing=timing,
                            gemini_options=gemini_options, errors=errors)
    else:
        video = [n for n in needed if n not in photo_nums]
        photo = [n for n in needed if n in photo_nums]
        if video:
            _video_step(video, segments=segments, cfg=cfg, transport=transport,
                        prompts_path=prompts_path, profile=profile, timing=timing,
                        gemini_options=gemini_options, errors=errors)
        if photo:
            _photo_step(photo, segments=segments, photo_profile=photo_profile, photo_cfg=photo_cfg,
                        transport=transport, prompts_path=prompts_path,
                        gemini_options=gemini_options, errors=errors)
        state = read_state(prompts_path)  # после фото: состояние файла с обоими видами
    prompts = state.setdefault("prompts", {})
    ready = [n for n in needed if _prompt_ready(prompts.get(str(n)))]
    return PromptRun(ready=ready, state=state, errors=errors)
