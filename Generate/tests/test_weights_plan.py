"""Список весов и тип профиля для setup.sh читаются из weights.json и profile.yaml через core/weights_plan.py."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

GEN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GEN / "core"))

import weights_plan as wp  # noqa: E402

MODELS = GEN / "model_profiles" / "models"
LTX = MODELS / "ltx25"
ZIMAGE = MODELS / "zimage_turbo"

LTX_BASE = "https://huggingface.co/Lightricks/LTX-2.5/resolve/main"
ZIMAGE_BASE = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main"


def test_ltx_rows_match_original_list_and_order():
    assert wp.file_rows(str(LTX)) == [
        ("diffusion_models", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "", LTX_BASE),
        ("text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "", LTX_BASE),
        ("vae", "ltx-2.5-video-vae-bf16.safetensors", "", LTX_BASE),
        ("vae", "ltx-2.5-audio-vae-bf16.safetensors", "", LTX_BASE),
        ("latent_upscale_models", "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "", LTX_BASE),
    ]


def test_zimage_rows_match_original_list():
    assert wp.file_rows(str(ZIMAGE)) == [
        ("diffusion_models", "z_image_turbo_bf16.safetensors",
         "split_files/diffusion_models/z_image_turbo_bf16.safetensors", ZIMAGE_BASE),
        ("text_encoders", "qwen_3_4b.safetensors",
         "split_files/text_encoders/qwen_3_4b.safetensors", ZIMAGE_BASE),
        ("vae", "ae.safetensors", "split_files/vae/ae.safetensors", ZIMAGE_BASE),
    ]


def test_zimage_download_url_uses_repo_path():
    row = wp.file_rows(str(ZIMAGE))[0]
    assert wp.download_url(row) == (
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
        "split_files/diffusion_models/z_image_turbo_bf16.safetensors"
    )


def test_ltx_download_url_without_repo_path():
    row = wp.file_rows(str(LTX))[0]
    assert wp.download_url(row) == (
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/"
        "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    )


def test_meta_ltx():
    assert wp.meta(str(LTX)) == {"type": "video", "needs_token": True, "files_total": 5}


def test_meta_zimage():
    assert wp.meta(str(ZIMAGE)) == {"type": "image", "needs_token": False, "files_total": 3}


def _write_weights(folder: pathlib.Path, files, base="https://example.invalid/base"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "weights.json").write_text(
        json.dumps({"base_url": base, "needs_token": False, "files": files}), encoding="utf-8"
    )


def test_missing_weights_json_raises(tmp_path):
    with pytest.raises(ValueError):
        wp.load_weights(str(tmp_path))


def test_empty_files_list_raises(tmp_path):
    _write_weights(tmp_path, [])
    with pytest.raises(ValueError):
        wp.load_weights(str(tmp_path))


def test_profile_type_audio_raises(tmp_path):
    (tmp_path / "profile.yaml").write_text("name: x\ntype: audio\n", encoding="utf-8")
    with pytest.raises(ValueError):
        wp.profile_type(str(tmp_path))


def test_no_type_line_means_video(tmp_path):
    (tmp_path / "profile.yaml").write_text("name: x\n", encoding="utf-8")
    assert wp.profile_type(str(tmp_path)) == "video"


def test_file_entry_without_name_raises(tmp_path):
    _write_weights(tmp_path, [{"subdir": "vae"}])
    with pytest.raises(ValueError):
        wp.file_rows(str(tmp_path))
