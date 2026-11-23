"""Офлайн-тесты I2V-графа cartoon_i2v, эндпоинта POST /input_image и разрешения профиля *_i2v."""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
import yaml

try:
    from Generate.worker import comfy_client as cc
    from Generate.worker import server as srv
    from Generate.worker.server import WorkerApp
except ImportError:  # pragma: no cover
    from worker import comfy_client as cc  # type: ignore
    from worker import server as srv  # type: ignore
    from worker.server import WorkerApp  # type: ignore

GEN = Path(__file__).resolve().parents[1]
I2V_DIR = GEN / "Nishes" / "cartoon" / "i2v"
WORKFLOW_PATH = I2V_DIR / "workflow.json"
BINDINGS_PATH = I2V_DIR / "bindings.json"
PROFILE_PATH = I2V_DIR / "profile.yaml"
WEIGHTS_PATH = I2V_DIR / "weights.json"
SETUP_SH = GEN / "worker" / "setup.sh"
BOOTSTRAP_SH = GEN / "worker" / "bootstrap.sh"

TOKEN = "SECRET-TOKEN-123"
WRONG = "WRONG-TOKEN-999"
VIDEO_BYTES = b"fake-mp4-data" * 16
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class FakeComfy:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.graphs: list[dict] = []
        self._n = 0

    def check_health(self) -> dict:
        return {"devices": [{"name": "FakeGPU", "vram_free": 12345, "vram_total": 99999}]}

    def queue_prompt(self, graph: dict) -> str:
        self.graphs.append(graph)
        return f"pid-{len(self.graphs)}"

    def wait_for_completion(self, prompt_id: str, timeout: float = 600.0, poll_interval: float = 1.0) -> str:
        self._n += 1
        sub = self.out_dir / "sub"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"clip_{self._n}.mp4").write_bytes(VIDEO_BYTES)
        return f"sub/clip_{self._n}.mp4"


@pytest.fixture
def make_i2v_app(tmp_path, monkeypatch):
    apps = []

    def factory(**kw):
        out = tmp_path / "out"
        inp = tmp_path / "input"
        out.mkdir(exist_ok=True)
        inp.mkdir(exist_ok=True)
        monkeypatch.setenv("COMFY_INPUT_DIR", str(inp))
        comfy = FakeComfy(out)
        workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
        kw.setdefault("idle_timeout_seconds", 3600)
        kw.setdefault("on_idle_timeout", lambda: None)
        kw.setdefault("log_dir", str(tmp_path / "logs"))
        app = WorkerApp(TOKEN, comfy, host="127.0.0.1", port=0, workflow=workflow,
                        bindings=bindings, output_dir=out, **kw)
        app.start_background()
        apps.append(app)
        return app, comfy, inp, out

    yield factory
    for a in apps:
        a.close()


def _client(app: WorkerApp, token: str | None = TOKEN) -> httpx.Client:
    headers = {"X-Worker-Token": token} if token else {}
    return httpx.Client(base_url=f"http://127.0.0.1:{app.port}", headers=headers,
                        trust_env=False, timeout=10)


def _wait_state(cli: httpx.Client, task_id: str, state: str, timeout: float = 5.0) -> dict:
    end = time.monotonic() + timeout
    body = {}
    while time.monotonic() < end:
        body = cli.get(f"/task/{task_id}").json()
        if body["state"] == state:
            return body
        time.sleep(0.02)
    raise AssertionError(f"Состояние {state} не достигнуто, последнее: {body}")


# ---------------------------------------------------------------------------
# Инварианты графа cartoon/i2v и bindings.json
# ---------------------------------------------------------------------------

def test_cartoon_i2v_workflow_and_bindings_invariants():
    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    weights = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    graph = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))

    assert set(bindings) == {"prompt", "num_frames", "seed", "filename_prefix", "input_image"}
    for key, spec in bindings.items():
        nid = str(spec["node_id"])
        field = str(spec["field"])
        assert nid in graph, f"missing node {nid} for {key}"
        assert field in graph[nid]["inputs"], f"missing field {field} on {nid}"

    # Каждая ссылка [node_id, slot] ведёт на существующий узел
    for nid, node in graph.items():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in graph, f"dangling link {v[0]} from {nid}.{k}"

    # Без встроенного улучшайзера промпта
    classes = {n.get("class_type") for n in graph.values()}
    assert "TextGenerateLTX2Prompt" not in classes
    assert "ComfySwitchNode" not in classes
    assert graph["398:364"]["inputs"]["text"] == [bindings["prompt"]["node_id"], 0]

    # Ровно 5 весов из weights.json / profile.yaml
    dir_by_node = {
        "UNETLoader": "diffusion_models",
        "CLIPLoader": "text_encoders",
        "VAELoader": "vae",
        "LatentUpscaleModelLoader": "latent_upscale_models",
    }
    key_by_node = {
        "UNETLoader": "unet_name",
        "CLIPLoader": "clip_name",
        "VAELoader": "vae_name",
        "LatentUpscaleModelLoader": "model_name",
    }
    graph_pairs = set()
    for node in graph.values():
        cls = node.get("class_type")
        if cls in dir_by_node:
            graph_pairs.add((dir_by_node[cls], node["inputs"][key_by_node[cls]]))
    weight_pairs = {(e["subdir"], e["name"]) for e in weights["files"]}
    assert len(weight_pairs) == 5
    assert graph_pairs == weight_pairs
    assert set(profile["model_files"].values()) == {e["name"] for e in weights["files"]}

    # SaveVideo = "75", LoadImage = "395", ImageScale 1920x1080 lanczos center, без аудио на выходе
    assert graph["75"]["class_type"] == "SaveVideo"
    assert graph["395"]["class_type"] == "LoadImage"
    scale_nodes = [n for n in graph.values() if n.get("class_type") == "ImageScale"]
    assert len(scale_nodes) == 1
    assert scale_nodes[0]["inputs"]["width"] == 1920
    assert scale_nodes[0]["inputs"]["height"] == 1080
    assert scale_nodes[0]["inputs"]["upscale_method"] == "lanczos"
    assert scale_nodes[0]["inputs"]["crop"] == "center"
    create_video = next(n for n in graph.values() if n.get("class_type") == "CreateVideo")
    assert "audio" not in create_video["inputs"]

    # Негативный промпт без cartoon / childish
    neg_text = graph["398:373"]["inputs"]["text"].lower()
    assert "cartoon" not in neg_text and "childish" not in neg_text


# ---------------------------------------------------------------------------
# inject_workflow с input_image
# ---------------------------------------------------------------------------

def test_inject_workflow_with_input_image():
    graph = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))

    out = cc.inject_workflow(
        graph, bindings, "cartoon scene prompt", 57, 123, "clip_42",
        second_seed=999, input_image="clip_42.png",
    )
    assert out["398:376"]["inputs"]["value"] == "cartoon scene prompt"
    assert out["398:378"]["inputs"]["value"] == 57
    assert out["398:339"]["inputs"]["noise_seed"] == 123
    assert out["398:338"]["inputs"]["noise_seed"] == 999
    assert out["75"]["inputs"]["filename_prefix"] == "clip_42"
    assert out["395"]["inputs"]["image"] == "clip_42.png"
    assert graph["395"]["inputs"]["image"] != "clip_42.png"


def test_inject_workflow_input_image_missing_binding_raises():
    graph = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
    bindings.pop("input_image")

    with pytest.raises(cc.ComfyError, match="input_image"):
        cc.inject_workflow(graph, bindings, "p", 25, 1, "clip_1", input_image="clip_1.png")


# ---------------------------------------------------------------------------
# POST /input_image: авторизация, валидация, лимит размера, атомарная запись
# ---------------------------------------------------------------------------

def test_post_input_image_auth_validation_and_atomic_write(make_i2v_app, monkeypatch):
    app, _, inp, _ = make_i2v_app()

    with _client(app, None) as c:
        assert c.post("/input_image?name=clip_1.png", content=PNG_BYTES).status_code == 401
    with _client(app, WRONG) as c:
        assert c.post("/input_image?name=clip_1.png", content=PNG_BYTES).status_code == 401

    with _client(app) as c:
        for bad_name in ("", "../evil.png", "a/b.png", "clip_.png", "clip_1.jpg", "other.png", "clip_1.png%0A"):
            resp = c.post(f"/input_image?name={bad_name}", content=PNG_BYTES)
            assert resp.status_code == 400, bad_name
        assert c.post("/input_image", content=PNG_BYTES).status_code == 400
        assert c.post("/input_image?name=clip_1.png", content=b"").status_code == 400

        # Превышение лимита размера -> 413
        monkeypatch.setattr(srv, "MAX_INPUT_IMAGE_BYTES", 10)
        resp_413 = c.post("/input_image?name=clip_1.png", content=b"x" * 11)
        assert resp_413.status_code == 413
        monkeypatch.undo()

        # Успешная запись
        ok = c.post("/input_image?name=clip_1.png", content=PNG_BYTES)
        assert ok.status_code == 200
        assert ok.json()["ok"] is True
        assert ok.json()["name"] == "clip_1.png"
        assert (inp / "clip_1.png").read_bytes() == PNG_BYTES
        assert list(inp.glob("*.tmp")) == []

        # Повторная запись перезаписывает атомарно
        new_bytes = PNG_BYTES + b"v2"
        assert c.post("/input_image?name=clip_1.png", content=new_bytes).status_code == 200
        assert (inp / "clip_1.png").read_bytes() == new_bytes
        assert list(inp.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# POST /task с input_image и удаление PNG после ack
# ---------------------------------------------------------------------------

def test_post_task_with_input_image_and_ack_cleanup(make_i2v_app):
    app, comfy, inp, out = make_i2v_app()
    base_task = {"num": 7, "prompt": "анимация кадра", "num_frames": 49, "seed": 42}

    with _client(app) as c:
        # Файл ещё не загружен -> 400
        assert c.post("/task", json={**base_task, "input_image": "clip_7.png"}).status_code == 400
        # Некорректное имя -> 400
        assert c.post("/task", json={**base_task, "input_image": "../clip_7.png"}).status_code == 400

        # Загружаем входной кадр и запускаем задачу
        assert c.post("/input_image?name=clip_7.png", content=PNG_BYTES).status_code == 200
        assert (inp / "clip_7.png").is_file()

        resp = c.post("/task", json={**base_task, "input_image": "clip_7.png"})
        assert resp.status_code == 200
        tid = resp.json()["task_id"]
        _wait_state(c, tid, "done")

        assert comfy.graphs[0]["395"]["inputs"]["image"] == "clip_7.png"
        assert comfy.graphs[0]["398:378"]["inputs"]["value"] == 49
        assert (inp / "clip_7.png").is_file()

        # Скачивание и ack -> удаляются и mp4, и временный PNG
        assert c.get(f"/file/{tid}").status_code == 200
        assert c.post(f"/ack/{tid}").status_code == 200
        assert not (inp / "clip_7.png").exists()
        assert list(out.rglob("*.mp4")) == []
        # Повторный ack (при уже удалённом файле) не падает
        assert c.post(f"/ack/{tid}").status_code == 200


# ---------------------------------------------------------------------------
# Разрешение профиля <niche>_i2v -> Nishes/<niche>/i2v
# ---------------------------------------------------------------------------

def test_find_profile_folder_resolves_i2v(tmp_path):
    models_root = GEN / "model_profiles" / "models"
    nishes_root = GEN / "Nishes"
    assert srv.find_profile_folder(models_root, nishes_root, "cartoon_i2v") == I2V_DIR

    # Проверка приоритета: models/<name> -> Nishes/<name> -> Nishes/<niche>/i2v
    m_root = tmp_path / "models"
    n_root = tmp_path / "Nishes"
    (n_root / "demo" / "i2v").mkdir(parents=True)
    assert srv.find_profile_folder(m_root, n_root, "demo_i2v") == n_root / "demo" / "i2v"

    (n_root / "demo_i2v").mkdir(parents=True)
    assert srv.find_profile_folder(m_root, n_root, "demo_i2v") == n_root / "demo_i2v"

    (m_root / "demo_i2v").mkdir(parents=True)
    assert srv.find_profile_folder(m_root, n_root, "demo_i2v") == m_root / "demo_i2v"


def test_shell_scripts_resolve_i2v_profile_dir():
    for path in (SETUP_SH, BOOTSTRAP_SH):
        text = path.read_text(encoding="utf-8")
        assert "*_i2v" in text and "%_i2v" in text, path.name
