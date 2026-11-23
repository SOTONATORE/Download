"""Офлайн-тесты I2V-оценки и двухэтапного конвейера промптов (score_i2v_segments, build_i2v_prompts)."""
from __future__ import annotations

import argparse
import json
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from Generate import cli
from Generate.core import gemini_prompts as gp
from Generate.core import prompt_check as pc
from Generate.core import prompt_pipeline as pp
from Generate.core import video_value as vv
from Generate.core.segment_split import Segment
from Generate.core.srt_parser import Segment as SrtSegment

GEN = Path(__file__).resolve().parents[1]
I2V_DIR = GEN / "Nishes" / "cartoon" / "i2v"
I2V_RULES = I2V_DIR / "prompts" / "rules.yaml"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("тест обратился к сети")

    monkeypatch.setattr(socket.socket, "connect", boom)


def test_cartoon_i2v_rules_yaml_flags():
    rules = gp.load_rules(I2V_RULES)
    assert rules.fire_hands_allowed is False  # fire_hands: forbidden
    assert rules.scene_motion_wording == "allow"
    assert pc.PROBLEM_FIRE_HANDS in {
        pc.problem_code(p)
        for p in pc.find_problems("The character reaches bare hands into open flames.", "video", rules=rules)
    }


def test_score_i2v_segments_skips_short_and_builds_prompt_with_photo():
    segs = [
        Segment(id=1, start=0.0, end=1.2, text="too short"),
        Segment(id=2, start=1.2, end=5.2, text="worker digs soil"),
        Segment(id=3, start=5.2, end=9.2, text="static checklist card"),
    ]
    photos = {
        1: "short photo",
        2: "A wide shot of the main character digging soil with a garden fork.",
        3: "A plain checklist card reading SOIL CHECK in bold black uppercase lettering.",
    }
    seen_prompts: list[str] = []

    def fake_send(prompt: str) -> str:
        seen_prompts.append(prompt)
        ids = [int(m) for m in re.findall(r"^id (\d+) \(", prompt, flags=re.M)]
        return json.dumps([{"id": i, "video_value": 90 if i == 2 else 15} for i in ids])

    values = vv.score_i2v_segments(segs, photos, fake_send, min_anim_sec=2.0)
    assert values == {2: 90, 3: 15}
    assert len(seen_prompts) == 1
    assert "id 1 (" not in seen_prompts[0]
    assert "worker digs soil" in seen_prompts[0]
    assert "Photo: A wide shot of the main character digging soil" in seen_prompts[0]


def test_score_i2v_segments_retries_then_raises_on_double_failure():
    segs = [Segment(id=1, start=0.0, end=4.0, text="action")]
    calls = {"n": 0}

    def bad_send(prompt: str) -> str:
        calls["n"] += 1
        return "not json"

    with pytest.raises(ValueError, match="I2V video_value"):
        vv.score_i2v_segments(segs, {1: "photo"}, bad_send, min_anim_sec=2.0)
    assert calls["n"] == vv.MAX_ATTEMPTS


def _make_i2v_cfgs(tmp_path: Path):
    photo_style = tmp_path / "photo_style.md"
    photo_style.write_text("# Style\nFlat 2D cartoon.", encoding="utf-8")
    preset = tmp_path / "preset.md"
    preset.write_text("Soft pastel cartoon finish.", encoding="utf-8")
    video_style = tmp_path / "video_style.md"
    video_style.write_text("# I2V Style\nAnimate gently.", encoding="utf-8")

    photo_cfg = gp.PromptConfig(
        style_file=str(photo_style),
        media="image",
        preset=gp.PRESET_NONE,
        niche_preset_file=str(preset),
        batch_size=15,
    )
    video_cfg = gp.PromptConfig(
        style_file=str(video_style),
        media="video",
        preset=gp.PRESET_NONE,
        batch_size=15,
        rules=gp.load_rules(I2V_RULES),
    )
    return photo_cfg, video_cfg, "Soft pastel cartoon finish."


def test_build_i2v_prompts_end_to_end_strips_preset_and_resumes(monkeypatch, tmp_path):
    photo_cfg, video_cfg, preset_tail = _make_i2v_cfgs(tmp_path)
    segments = [
        SrtSegment(num=1, start_ms=0, end_ms=1500, text="Short segment."),       # 1.5s < 2.0s -> только фото
        SrtSegment(num=2, start_ms=1500, end_ms=6500, text="Hero walks forward."), # 5.0s -> кандидат
        SrtSegment(num=3, start_ms=6500, end_ms=10500, text="Checklist card."),    # 4.0s -> кандидат
    ]
    prompts_path = tmp_path / "prompts.json"
    dist_path = tmp_path / "distribution.json"
    timing = SimpleNamespace(
        fps=24,
        frame_rule={"kind": "8k+1"},
        clip_min_sec=2.0,
        clip_max_sec=8.0,
        gpu_max_sec=10.0,
        tail_pad_sec=0.5,
    )

    video_calls: list[dict] = []

    def fake_build_prompts(needed, *, segments, photo_nums, cfg, profile, timing, photo_profile, photo_cfg, transport,
                           gemini_options, prompts_path):
        state = {
            "srt_hash": "h1",
            "model": "m",
            "profile": profile,
            "prompts": {
                str(n): {
                    "num": n,
                    "num_frames": 1,
                    "prompt": f"A clean cartoon frame {n}. {preset_tail}",
                    "status": "ready",
                    "attempts": 1,
                    "error": "",
                    "needs_review": False,
                    "extra": {},
                }
                for n in needed
            },
        }
        Path(prompts_path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return pp.PromptRun(ready=list(needed), state=state, errors=[], abort=None)

    def fake_generate_prompts(segs, wanted, cfg, managed, model, on_batch=None, **kw):
        video_calls.append({
            "wanted": list(wanted),
            "source_photo_prompts": dict(cfg.source_photo_prompts or {}),
            "segment_durations": dict(cfg.segment_durations or {}),
        })
        results = [
            gp.PromptResult(
                num=n,
                prompt=f"The character moves calmly across frame {n}.",
                needs_review=False,
                review_reason="",
            )
            for n in wanted
        ]
        if on_batch:
            on_batch(results)
        return results

    monkeypatch.setattr(pp, "build_prompts", fake_build_prompts)
    monkeypatch.setattr(pp.gp, "generate_prompts", fake_generate_prompts)
    monkeypatch.setattr(pp.go, "ManagedTransport", lambda *a, **k: SimpleNamespace(active_model="m"))

    score_calls = {"n": 0}

    def fake_score_send(prompt: str) -> str:
        score_calls["n"] += 1
        ids = [int(m) for m in re.findall(r"^id (\d+) \(", prompt, flags=re.M)]
        # Оцениваем #2 высоко, #3 низко
        return json.dumps([{"id": i, "video_value": 95 if i == 2 else 10} for i in ids])

    run = pp.build_i2v_prompts(
        [1, 2, 3],
        segments=segments,
        photo_cfg=photo_cfg,
        photo_profile="cartoon",
        video_cfg=video_cfg,
        video_profile="cartoon_i2v",
        timing=timing,
        transport=SimpleNamespace(),
        gemini_options={},
        prompts_path=prompts_path,
        distribution_path=dist_path,
        intro_seconds=180.0,
        intro_video_percent=70.0,
        after_intro_video_percent=10.0,
        min_anim_sec=2.0,
        score_send=fake_score_send,
    )

    assert run.abort is None
    assert run.ready == [1, 2, 3]
    assert score_calls["n"] == 1
    assert dist_path.is_file()
    assert len(video_calls) == 1
    # Сегмент 1 (1.5 с) не мог попасть в видео; 2 выбран, 3 отсеян по квоте/оценке
    selected = video_calls[0]["wanted"]
    assert 1 not in selected
    assert selected == [2]
    # Хвост preset.md снят с source_photo_prompts
    src = video_calls[0]["source_photo_prompts"][2]
    assert preset_tail not in src
    assert src == "A clean cartoon frame 2"
    assert video_calls[0]["segment_durations"][2] == pytest.approx(5.0)

    saved = json.loads(prompts_path.read_text(encoding="utf-8"))
    assert saved["i2v_mode"] is True
    assert saved["video_profile"] == "cartoon_i2v"
    assert saved["prompts"]["1"]["i2v"] is False
    assert saved["prompts"]["2"]["i2v"] is True
    assert saved["prompts"]["2"]["video_status"] == "ready"
    assert saved["prompts"]["2"]["video_prompt"].startswith("The character moves calmly")
    nf = saved["prompts"]["2"]["video_num_frames"]
    assert isinstance(nf, int) and nf > 0 and (nf - 1) % 8 == 0
    assert saved["prompts"]["2"]["video_needs_review"] is False

    # Повторный вызов (resume): уже сохранённый выбор i2v не пересчитывается и LLM не вызывается
    video_calls.clear()
    score_calls["n"] = 0

    def resume_build_prompts(needed, *, prompts_path, **kw):
        st = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
        return pp.PromptRun(ready=list(needed), state=st, errors=[], abort=None)

    monkeypatch.setattr(pp, "build_prompts", resume_build_prompts)

    def fail_score_send(_prompt: str) -> str:
        raise AssertionError("score_send не должен вызываться при resume с сохранённым выбором i2v")

    resumed = pp.build_i2v_prompts(
        [1, 2, 3],
        segments=segments,
        photo_cfg=photo_cfg,
        photo_profile="cartoon",
        video_cfg=video_cfg,
        video_profile="cartoon_i2v",
        timing=timing,
        transport=SimpleNamespace(),
        gemini_options={},
        prompts_path=prompts_path,
        score_send=fail_score_send,
    )
    assert resumed.ready == [1, 2, 3]
    assert score_calls["n"] == 0
    assert video_calls == []


def test_prepare_prompts_only_i2v_inspects_photo_and_video_prompts(monkeypatch, tmp_path):
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nFarmer digs soil.\n\n"
        "2\n00:00:05,000 --> 00:00:09,000\nChart on wall.\n",
        encoding="utf-8",
    )
    missing_file = tmp_path / "missing.txt"
    missing_file.write_text("1\n2\n", encoding="utf-8")
    prompts_path = tmp_path / "prompts.json"
    art_dir = tmp_path / "art"
    art_dir.mkdir()

    def fake_build_i2v(needed, **kw):
        # Видео для сегмента 1: корректное число кадров 8k+1 = 137 (5.0s + 0.5s tail * 24fps = 132 -> 137)
        state = {
            "i2v_mode": True,
            "video_profile": "cartoon_i2v",
            "prompts": {
                "1": {
                    "num": 1,
                    "num_frames": 1,
                    "prompt": "A wide shot of one adult farmer digging soil.",
                    "status": "ready",
                    "needs_review": False,
                    "error": "",
                    "i2v": True,
                    "video_prompt": "The farmer lifts the fork steadily.",
                    "video_num_frames": 137,
                    "video_status": "ready",
                    "video_needs_review": True,
                    "video_error": "тестовая пометка видео",
                },
                "2": {
                    "num": 2,
                    "num_frames": 1,
                    "prompt": "A flat chart on a cream wall.",
                    "status": "ready",
                    "needs_review": False,
                    "error": "",
                    "i2v": False,
                },
            },
        }
        Path(kw["prompts_path"]).write_text(json.dumps(state), encoding="utf-8")
        return pp.PromptRun(ready=[1, 2], state=state, errors=[], abort=None)

    monkeypatch.setattr(pp, "build_i2v_prompts", fake_build_i2v)
    monkeypatch.setattr(cli, "_make_transport", lambda cfg, family: SimpleNamespace())

    ns = argparse.Namespace(
        srt=str(srt_file),
        missing=str(missing_file),
        prompts_path=str(prompts_path),
        artifact_dir=str(art_dir),
        video_model="cartoon_i2v",
        video_profile="cartoon_i2v",
        photo_profile="cartoon",
        video_percent="0",
        intro_seconds="180",
        limit_clips=0,
        preset="none",
        text_model="gemini",
        fail_fast=False,
        i2v_mode=True,
        intro_minutes=3.0,
        intro_video_percent=70.0,
        after_intro_video_percent=10.0,
        min_anim_sec=2.0,
    )
    prep = cli.prepare_prompts_only(ns)
    assert prep.photo_nums == [1, 2]
    assert prep.video_nums == [1]
    assert prep.ready_photo == 2
    assert prep.ready_video == 1
    assert "видео клип 1: на проверку" in prep.problems
    assert "тестовая пометка видео" in prep.report_md
    assert "Режим I2V: включён" in prep.report_md
