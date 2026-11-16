"""Самостоятельный A/B-тест сценариев workflow ComfyUI (ниша cartoon_photos).

Один арендованный RTX 5090 прогоняет 16 сцен (4 варианта × 4 разных сюжета) через SSH-туннель к ComfyUI.
PNG и служебные файлы пишутся только в каталог артефакта запуска GitHub Actions. Релизы (Release) не
создаются и не изменяются. Оркестратор генерации (orchestrator.py) не используется: здесь нет сетки
карт, Release и весов профиля, поэтому путь намеренно короче.

Варианты (см. SCENE_CATALOG):
  A  pure_t2i_no_refs           — без опорных изображений, без VAE в узле 491, латент 1920x1088.
  B  hybrid_ref6_with_vae       — герои (B1, B2): ref_6 + VAE, латент из узла 491; остальные — как A.
  C  ref6_vlm_only_no_vae       — герои (C1, C2): ref_6 без VAE, латент пустой; остальные — как A.
  D  style_vlm_only_no_vae      — всегда без VAE, пустой латент; опорные: ref_1 и/или ref_2, ref_6.

Журнал и комментарии — на русском; тексты промптов, которые видит модель, — на английском.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import httpx
    from Generate.core import gemini_prompts as gp
    from Generate.core import ssh_probe
    from Generate.core.bad_hosts import BadHostsError, load_bad_hosts
    from Generate.core.budget import BudgetError, BudgetTracker
    from Generate.core.log_setup import (mask_external, mask_fetched_text, mask_text, secret_mask_values,
                                         set_known_secrets)
    from Generate.core.orchestrator import (
        COMFY_ANSWER_LIMIT_SEC, COMFYUI_ARGS, NOT_FOUND_LIMIT, STOPPED_AFTER_START_STATUSES, _has_start_marker,
        _internal_ports_view, _valid_worker_build, _worker_onstart)
    from Generate.core.vast_client import (OfferFilter, VastError, VastInstanceNotFound,
                                           remove_instance_fields)
except ImportError:  # pragma: no cover - плоский импорт из каталога core
    import httpx  # type: ignore
    import gemini_prompts as gp  # type: ignore
    import ssh_probe  # type: ignore
    from bad_hosts import BadHostsError, load_bad_hosts  # type: ignore
    from budget import BudgetError, BudgetTracker  # type: ignore
    from log_setup import (mask_external, mask_fetched_text, mask_text, secret_mask_values,  # type: ignore
                           set_known_secrets)
    from orchestrator import (  # type: ignore
        COMFY_ANSWER_LIMIT_SEC, COMFYUI_ARGS, NOT_FOUND_LIMIT, STOPPED_AFTER_START_STATUSES, _has_start_marker,
        _internal_ports_view, _valid_worker_build, _worker_onstart)
    from vast_client import (OfferFilter, VastError, VastInstanceNotFound,  # type: ignore
                             remove_instance_fields)


log = logging.getLogger("Generate.workflow_ab_test")

# --- Константы стенда ---------------------------------------------------------------------------

NICHE = "cartoon_photos"
MODEL_PROFILE = NICHE
TESTWF_IMAGE = "vastai/comfy:v0.41.0-cuda-13.2-py312"  # образ как у стокового пути (config.default.yaml)
NICHE_DIR = Path(__file__).resolve().parents[1] / "Nishes" / NICHE
WORKFLOW_PATH = NICHE_DIR / "workflow.json"
BINDINGS_PATH = NICHE_DIR / "bindings.json"

COMFY_REMOTE_PORT = 18188        # порт ComfyUI внутри карты (COMFYUI_ARGS: --port 18188)
WORKER_HTTP_PORT = 8000          # внешний порт воркера, по которому идёт /health
LOGS_TAIL_LINES = 2000           # строк журнала контейнера при запросе логов снимаемой карты через Vast
AFFORD_ESTIMATE_SEC = 1800.0     # время, на которое проверяется остаток бюджета перед арендой (как в can_afford)
TUNNEL_READY_SEC = 30.0          # ожидание открытия SSH-туннеля
KEEPALIVE_SEC = 10.0             # пинг /health, чтобы DeadManSwitch воркера не сработал
POLL_SEC = 15.0                  # опрос статуса карты и /health до готовности
SCENE_POLL_SEC = 5.0             # опрос /history/{id} во время рендера
SCENE_TIMEOUT_SEC = 900.0       # потолок на один сценарий (первый рендер грузит веса в память)
HTTP_TIMEOUT_SEC = 30.0

# Фиксированное зерно: все 16 сцен одинаковы по семени, поэтому разница между вариантами — только в графе.
TESTWF_SEED = 679152989211317

LATENT_W = 1920
LATENT_H = 1088                  # кратно 16, как у узла VAE; узел 511 масштабирует результат в 1920x1080

# Узлы шаблона workflow.json (см. bindings.json)
NODE_TEXT = "491"                # TextEncodeQwenImage21
NODE_SAMPLER = "488"             # KSampler
NODE_VAE_LOADER = "497"          # VAELoader
NODE_SAVE = "461"                # SaveImageAdvanced
NODE_EMPTY_LATENT = "512"        # новый узел EmptyLatentImage (1920x1088), не пересекается с шаблоном (макс. 511)
REF_1_NODE = "470"               # ref_1.png
REF_2_NODE = "475"               # ref_2.png

VARIANT_NAMES = {
    "A": "pure_t2i_no_refs",
    "B": "hybrid_ref6_with_vae",
    "C": "ref6_vlm_only_no_vae",
    "D": "style_vlm_only_no_vae",
}

# --- Каталог сцен (16 уникальных, без повторов сюжета между вариантами) --------------------------

_PROMPT_A1 = ("A young man with a round bald white head, black stick-figure eyes and mouth, wearing a bright blue "
              "short-sleeved t-shirt, dark blue jeans, and white sneakers, running frantically along a dark forest "
              "trail at night. He holds a blazing wooden torch in his right hand, looking back over his shoulder with "
              "wide terrified eyes and an open screaming mouth. Tall dark green trees with thick black outlines, flat "
              "dark brown soil. Flat 2D cartoon style, bold dark outlines, solid flat color fills, simple shapes, no "
              "gradients, no shading, no 3D volume.")
_PROMPT_A2 = ("A young man with a round bald white head, simple cartoon dot eyes, wearing a bright blue short-sleeved "
              "t-shirt, blue jeans, and white sneakers, sitting on the edge of a high grey stone cliff over the sea. He "
              "laughs happily with an open cheerful smile, pointing his left arm up toward large handwritten black "
              "capital text NEW WORLD centered in the bright sky blue sky. Bold dark outlines around every shape, flat "
              "color fills, clean 2D vector style, no gradients, no shadows.")
_PROMPT_A3 = ("Underwater scene in a calm shallow sea with bright turquoise water. In the center, a flat rocky seafloor "
              "with two dark grey segmented trilobites crawling on yellow sand. A small cluster of bright green sea "
              "anemones with thick outlines on the left. High contrast, clean bold black outlines around all creatures "
              "and rocks, solid flat color fills, simple readable shapes, one static frame, no people.")
_PROMPT_A4 = ("Plain solid white background filling the whole 16:9 frame. In the center, large bold handwritten capital "
              "text in two lines: LIMITED TIME in black letters, and OFFER in bright red letters. To the left of the "
              "text, a small flat heraldic shield colored solid cobalt blue with a thick black outline, positioned at "
              "the same height as the text. Crisp 2D graphic style, bold outlines, flat colors, no shadows, no "
              "hourglass, no people.")
_PROMPT_B1 = ("Draw the character from <image1> (round white head, blue t-shirt, jeans, white shoes), completely replacing "
              "the white background with a full landscape. The character is climbing up a steep grey rocky cliff covered "
              "in patches of white snow, seen in side view, facing right. His face has a determined, focused expression "
              "with furrowed brow and firm mouth. Behind him, jagged snow-covered mountain peaks under an overcast pale "
              "grey sky. Bold dark outlines, solid flat color fills, one static frame.")
_PROMPT_B2 = ("Draw the character from <image1> (round white head, blue t-shirt, jeans, white shoes), completely replacing "
              "the white background with an indoor scene. The character sits at a sturdy brown wooden desk in a simple "
              "laboratory, looking with wide surprised eyes at a glowing green chemical flask. Above the desk on the beige "
              "wall, large handwritten black capital letters read FIRST CLUE. Bold dark outlines, flat color fills, no "
              "gradients, one static frame.")
_PROMPT_B3 = ("A vast canyon landscape on a desolate desert planet under a dusty pale orange sky. Three large red rock arches "
              "cross the middle ground, with flat orange-brown sand and jagged dark red rock shelves in the foreground. "
              "Clean bold black outlines around every geological formation, solid flat color fills, simple geometric "
              "shapes, one static frame, no people.")
_PROMPT_B4 = ("Plain solid white background filling the entire frame. In the center, a bold horizontal black timeline arrow "
              "pointing to the right. Above the middle of the arrow, handwritten capital letters read 3 MILLION YEARS, with "
              "the word MILLION written in bright red and the rest in black. Below the arrow, three evenly spaced solid "
              "black dots. Clean 2D graphic layout, bold lines, flat colors, no people.")
_PROMPT_C1 = ("Draw the character shown in <image1> (round white head, blue t-shirt, jeans, white sneakers) walking across a "
              "wide flat green meadow in heavy rain. He hunches forward, covering his head with both arms against the "
              "falling rain, with an exhausted, unhappy facial expression. In the distance, dark grey rain clouds under a "
              "gloomy sky. Flat 2D cartoon style, bold dark outlines around every shape, flat color fills, no gradients, "
              "no 3D effects.")
_PROMPT_C2 = ("Draw the character shown in <image1> (round white head, blue t-shirt, jeans, white sneakers) standing on the "
              "wooden deck of a sailing ship, facing right. He holds a golden brass spyglass to his right eye, smiling "
              "happily. Centered in the bright sky blue sky above him, large handwritten black capital letters read LAND "
              "AHEAD. Bold black outlines, flat solid color fills, simple shapes, one static frame.")
_PROMPT_C3 = ("A dark volcanic island in the middle of a dark grey choppy sea. A conical black volcano rises in the center with "
              "a bright orange river of lava flowing down its slope into the water, and a billowing grey ash cloud in the "
              "sky. Flat 2D cartoon style, bold dark outlines, vibrant solid color fills, no gradients, one static frame, "
              "no people.")
_PROMPT_C4 = ("Plain solid white background filling the whole frame. In the center, large neat handwritten black capital letters "
              "read SPONSORED BY SURFSHARK. Immediately to the left of the phrase, a small flat silhouette of a blue shark "
              "swimming horizontally, drawn with a bold black outline, level with the middle of the text line. Minimalist "
              "2D graphic card, solid colors, no hourglass, no extra objects.")
_PROMPT_D1 = ("Draw the character from <image2> using ONLY the 2D line thickness and flat color palette of <image1>. Do not "
              "copy the ground, leaves, or landscape layers of <image1>. The character sneaks cautiously through a dense "
              "tropical jungle of giant green palm leaves at night, holding a silver flashlight in both hands, casting a "
              "flat yellow cone of light forward, with an alert, tense sideways glance. Bold dark outlines, solid flat "
              "colors, one static frame.")
_PROMPT_D2 = ("Draw the character from <image2> in the 2D illustration style of <image1> (thick dark outlines, flat color fills, "
              "no shading). Do not copy the soil or background of <image1>. The character sits on a thick brown fallen log "
              "beside a warm campfire at night, warming his open hands at a safe distance from the fire, looking relaxed and "
              "content. In the dark navy blue sky above, white handwritten capital letters read WARMTH. Bold outlines, flat "
              "colors.")
_PROMPT_D3 = ("A two-panel split frame in the visual style of <image1>, separated down the middle by a vertical dashed black line. "
              "Left panel: dry cracked red-brown earth under an empty pale sky, with the word DROUGHT in red handwritten "
              "capitals. Right panel: lush green valley with a roaring blue river under a cloudy sky, with the word FLOOD in "
              "green handwritten capitals. Bold dark outlines around each panel, flat colors, one static frame, no people.")
_PROMPT_D4 = ("Take the handwritten black capital font style of <image1> for the phrase CHAPTER TWO: ICE AGE, written on a plain "
              "solid white background filling the whole frame, centered horizontally. To the left of the phrase, a single "
              "flat light-blue chunk of iceberg with a bold black outline, level with the text line. Clean flat 2D graphic "
              "style, bold outlines, flat colors, no hourglass, no people.")


@dataclass(frozen=True)
class Scene:
    scene_id: str
    variant: str                     # "A" | "B" | "C" | "D"
    hero: bool                       # герой с опорным ref_6 (для B и C)
    refs: tuple[str, ...]            # имена опорных файлов в порядке image_1, image_2, ...
    prompt: str


SCENE_CATALOG: tuple[Scene, ...] = (
    Scene("A1_forest_torch", "A", False, (), _PROMPT_A1),
    Scene("A2_cliff_new_world", "A", False, (), _PROMPT_A2),
    Scene("A3_coral_reef", "A", False, (), _PROMPT_A3),
    Scene("A4_offer_card", "A", False, (), _PROMPT_A4),
    Scene("B1_snow_cliff", "B", True, ("ref_6",), _PROMPT_B1),
    Scene("B2_lab_desk", "B", True, ("ref_6",), _PROMPT_B2),
    Scene("B3_mars_canyon", "B", False, (), _PROMPT_B3),
    Scene("B4_timeline_card", "B", False, (), _PROMPT_B4),
    Scene("C1_rain_meadow", "C", True, ("ref_6",), _PROMPT_C1),
    Scene("C2_ship_spyglass", "C", True, ("ref_6",), _PROMPT_C2),
    Scene("C3_volcano_island", "C", False, (), _PROMPT_C3),
    Scene("C4_surfshark_card", "C", False, (), _PROMPT_C4),
    Scene("D1_jungle_flashlight", "D", False, ("ref_1", "ref_6"), _PROMPT_D1),
    Scene("D2_campfire_night", "D", False, ("ref_1", "ref_6"), _PROMPT_D2),
    Scene("D3_drought_flood", "D", False, ("ref_1",), _PROMPT_D3),
    Scene("D4_ice_age_card", "D", False, ("ref_2",), _PROMPT_D4),
)


class ABInputError(Exception):
    """Ошибка входных данных теста (код 2): проверяется до аренды карты."""


class ABRuntimeError(Exception):
    """Ошибка выполнения теста (код 3): карта, туннель, ComfyUI или запись артефакта."""


class GraphError(ABInputError):
    """Граф сцены не проходит проверку ссылок (ошибка сборки, до аренды)."""


# --- Сборка графов ------------------------------------------------------------------------------

def load_template(workflow_path: Path = WORKFLOW_PATH, bindings_path: Path = BINDINGS_PATH) -> tuple[dict, dict]:
    """Читает шаблон графа и bindings. Возвращает (граф, {ref_N: id узла LoadImage}).

    Проверяет, что каждый ref_N из bindings указывает на LoadImage с файлом ref_N.png: иначе шаблон
    расходится с каталогом сцен и тест не должен стартовать.
    """
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    ref_nodes: dict[str, str] = {}
    for name, spec in bindings.items():
        if not name.startswith("ref_"):
            continue
        node_id = str(spec["node_id"])
        node = workflow.get(node_id)
        if node is None or node.get("class_type") != "LoadImage":
            raise GraphError(f"В шаблоне нет LoadImage для {name} (узел {node_id}).")
        if node["inputs"].get("image") != f"{name}.png":
            raise GraphError(f"Узел {node_id} не указывает на файл {name}.png.")
        ref_nodes[name] = node_id
    return workflow, ref_nodes


def _base(template: dict, scene: Scene, seed: int) -> dict:
    """Копия шаблона с текстом сцены, зерном и префиксом имени файла. Шаблон не меняется."""
    graph = copy.deepcopy(template)
    graph[NODE_TEXT]["inputs"]["prompt"] = scene.prompt
    graph[NODE_SAMPLER]["inputs"]["seed"] = seed
    graph[NODE_SAVE]["inputs"]["filename_prefix"] = f"testwf_{scene.scene_id}"
    return graph


def _strip_loads(graph: dict) -> None:
    """Удаляет все LoadImage и подключения images.image_* к узлу 491."""
    for node_id in [k for k, v in graph.items() if v.get("class_type") == "LoadImage"]:
        del graph[node_id]
    inputs = graph[NODE_TEXT]["inputs"]
    for key in [k for k in inputs if k.startswith("images.image_")]:
        del inputs[key]


def _attach_refs(graph: dict, template: dict, ref_nodes: dict, refs: tuple[str, ...]) -> None:
    """Подключает опорные изображения по порядку к images.image_1..N узла 491 (LoadImage берутся из шаблона)."""
    for index, name in enumerate(refs, start=1):
        node_id = ref_nodes[name]
        graph[node_id] = copy.deepcopy(template[node_id])
        graph[NODE_TEXT]["inputs"][f"images.image_{index}"] = [node_id, 0]


def _drop_vae(graph: dict) -> None:
    graph[NODE_TEXT]["inputs"].pop("vae", None)


def _empty_latent(graph: dict) -> None:
    """Пустой латент 1920x1088 вместо выхода узла 491; KSampler читает его из нового узла."""
    graph[NODE_EMPTY_LATENT] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": LATENT_W, "height": LATENT_H, "batch_size": 1},
        "_meta": {"title": "Empty Latent Image (testwf)"},
    }
    graph[NODE_SAMPLER]["inputs"]["latent_image"] = [NODE_EMPTY_LATENT, 0]


def _build_pure_t2i(template: dict, scene: Scene, seed: int, ref_nodes: dict) -> dict:
    """Вариант A (и запасной путь для не-героев B и C): 0 опорных, без VAE в 491, пустой латент."""
    graph = _base(template, scene, seed)
    _strip_loads(graph)
    _drop_vae(graph)
    _empty_latent(graph)
    return graph


def _build_hybrid(template: dict, scene: Scene, seed: int, ref_nodes: dict) -> dict:
    """Вариант B: герой — ref_6 с VAE, латент берётся из узла 491 (слот 2); не герой — как A."""
    if not scene.hero:
        return _build_pure_t2i(template, scene, seed, ref_nodes)
    graph = _base(template, scene, seed)
    _strip_loads(graph)
    _attach_refs(graph, template, ref_nodes, scene.refs)
    graph[NODE_TEXT]["inputs"]["resolution"] = 0
    graph[NODE_SAMPLER]["inputs"]["latent_image"] = [NODE_TEXT, 2]
    return graph  # VAE в 491 сохраняется из шаблона


def _build_vlm_hero(template: dict, scene: Scene, seed: int, ref_nodes: dict) -> dict:
    """Вариант C: герой — ref_6 без VAE, пустой латент; не герой — как A."""
    if not scene.hero:
        return _build_pure_t2i(template, scene, seed, ref_nodes)
    graph = _base(template, scene, seed)
    _strip_loads(graph)
    _attach_refs(graph, template, ref_nodes, scene.refs)
    _drop_vae(graph)
    _empty_latent(graph)
    return graph


def _build_style_vlm(template: dict, scene: Scene, seed: int, ref_nodes: dict) -> dict:
    """Вариант D: всегда без VAE и с пустым латентом; опорные — по сцене (ref_1, ref_2, ref_6)."""
    graph = _base(template, scene, seed)
    _strip_loads(graph)
    _attach_refs(graph, template, ref_nodes, scene.refs)
    _drop_vae(graph)
    _empty_latent(graph)
    return graph


BUILDERS = {
    "A": _build_pure_t2i,
    "B": _build_hybrid,
    "C": _build_vlm_hero,
    "D": _build_style_vlm,
}


def build_graph(template: dict, ref_nodes: dict, scene: Scene, seed: int = TESTWF_SEED) -> dict:
    """Граф ComfyUI для одной сцены. Шаблон не меняется; результат проверяется validate_graph."""
    graph = BUILDERS[scene.variant](template, scene, seed, ref_nodes)
    validate_graph(graph)
    return graph


def validate_graph(graph: dict) -> None:
    """Проверяет ссылки: каждый вход-ссылка [id, слот] указывает на существующий узел, слот — неотрицательное целое.

    Бросает GraphError с первой найденной проблемой. Проверка не заменяет запуск ComfyUI, а отсекает
    битые сборки до аренды карты.
    """
    for node_id, node in graph.items():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                target, slot = value
                if target not in graph:
                    raise GraphError(f"Узел {node_id}.{key} ссылается на отсутствующий узел {target}.")
                if not isinstance(slot, int) or slot < 0:
                    raise GraphError(f"Узел {node_id}.{key}: неверный номер слота {slot!r}.")
    sampler = graph.get(NODE_SAMPLER)
    if sampler is None or "latent_image" not in sampler.get("inputs", {}):
        raise GraphError("В графе нет KSampler с latent_image.")


def graph_summary(graph: dict) -> dict:
    """Краткое описание графа для summary и тестов: опорные слоты, наличие VAE, источник латента."""
    inputs = graph[NODE_TEXT]["inputs"]
    refs = sorted(k for k in inputs if k.startswith("images.image_"))
    latent_src = graph[NODE_SAMPLER]["inputs"]["latent_image"]
    return {
        "refs": [(k, inputs[k][0]) for k in refs],
        "has_vae": "vae" in inputs,
        "latent_from": latent_src[0],
        "latent_slot": latent_src[1],
    }


# --- Поиск предложения и опрос карты ------------------------------------------------------------

def _offer_filter(cfg) -> OfferFilter:
    """Фильтр предложений из конфига (те же поля, что у cmd_run; без пилотных исключений)."""
    return OfferFilter(
        gpu_name=cfg.gpu_name or None,
        min_price=cfg.min_price_per_hour,
        max_price=cfg.max_price_per_hour,
        min_reliability=cfg.min_reliability,
        min_inet_mbps=cfg.min_inet_mbps,
        max_inet_price_per_tb=cfg.max_inet_price_per_tb,
        interruptible=cfg.interruptible,
        verified=cfg.verified,
        min_disk_gb=float(cfg.disk_gb),
        min_cuda_max=float(cfg.min_cuda_max),
        min_disk_bw_mbps=float(cfg.min_disk_bw_mbps),
        exclude_countries=list(cfg.exclude_countries),
    )


def _probe_health(url: str, token: str) -> Optional[dict]:
    """GET /health воркера с токеном. None, если запрос не дошёл или ответ не JSON-объект."""
    try:
        resp = httpx.get(url + "/health", headers={"X-Worker-Token": token}, timeout=HTTP_TIMEOUT_SEC)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class _Keepalive(threading.Thread):
    """Фоновый пинг GET /health каждые KEEPALIVE_SEC, чтобы DeadManSwitch воркера не сработал во время рендера.

    Ошибки связи не прерывают поток: они пишутся в журнал, а следующий пинг идёт по таймеру.
    """

    def __init__(self, url: str, token: str, interval: float = KEEPALIVE_SEC) -> None:
        super().__init__(name="testwf-keepalive", daemon=True)
        self._url = url
        self._token = token
        self._interval = interval
        self._halt = threading.Event()
        self.pings = 0
        self.failures = 0

    def run(self) -> None:
        while not self._halt.wait(self._interval):
            if _probe_health(self._url, self._token) is None:
                self.failures += 1
                log.warning("Пинг /health для keepalive не прошёл (всего сбоев: %d).", self.failures)
            else:
                self.pings += 1

    def stop(self) -> None:
        self._halt.set()


# --- Артефакт -----------------------------------------------------------------------------------

class _Artifact:
    """Запись файлов в каталог артефакта. Без каталога (пустой путь) все записи молча пропускаются."""

    def __init__(self, root: Optional[str]) -> None:
        self.root = Path(root) if root else None

    def write_text(self, rel: str, text: str) -> None:
        if self.root is None:
            return
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write_bytes(self, rel: str, data: bytes) -> None:
        if self.root is None:
            return
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def append_text(self, rel: str, text: str) -> None:
        if self.root is None:
            return
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)


INDEX_HEADER = "scene\tvariant\thero\tseed\tduration_sec\tstatus\tpng\trefs\tlatent_from\thas_vae\tdetail\tprompt\n"


@dataclass
class SceneResult:
    scene_id: str
    variant: str
    hero: bool
    seed: int
    prompt: str
    status: str = "not_run"          # ok | error | timeout | not_run
    duration_sec: float = 0.0
    png: str = ""
    detail: str = ""
    refs: str = ""
    latent_from: str = ""
    has_vae: bool = False

    def index_line(self) -> str:
        flat_prompt = self.prompt.replace("\t", " ").replace("\n", " ")
        detail = self.detail.replace("\t", " ").replace("\n", " ")
        return (f"{self.scene_id}\t{self.variant}\t{int(self.hero)}\t{self.seed}\t{self.duration_sec:.1f}\t"
                f"{self.status}\t{self.png}\t{self.refs}\t{self.latent_from}\t{int(self.has_vae)}\t"
                f"{detail}\t{flat_prompt}\n")


@dataclass
class ABRunSummary:
    results: list[SceneResult] = field(default_factory=list)
    instance_id: Optional[int] = None
    label: str = ""

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def ok(self) -> bool:
        return len(self.results) == len(SCENE_CATALOG) and self.ok_count == len(SCENE_CATALOG)


# --- Сцены через ComfyUI ------------------------------------------------------------------------

def _collect_images(outputs: dict) -> list[dict]:
    """Список картинок из history: сначала узел сохранения 461, затем любые узлы с полем images."""
    preferred = outputs.get(NODE_SAVE, {}).get("images", [])
    if preferred:
        return list(preferred)
    found: list[dict] = []
    for node_out in outputs.values():
        found.extend(node_out.get("images", []) or [])
    return found


def _error_text(status: dict) -> str:
    """Короткая причина ошибки из status.messages ComfyUI; текст маскируется на случай секретов."""
    messages = status.get("messages", []) or []
    for item in messages:
        if isinstance(item, list) and len(item) == 2 and item[0] == "execution_error":
            data = item[1] if isinstance(item[1], dict) else {}
            return mask_text(f"{data.get('exception_type', 'error')}: {str(data.get('exception_message', ''))[:200]}")
    return mask_text(str(messages)[:200]) if messages else "error"


def _render_scene(client: httpx.Client, scene: Scene, graph: dict, artifact: _Artifact,
                  stop: Optional[threading.Event], clock=time.monotonic, sleep=time.sleep) -> SceneResult:
    """Ставит граф в очередь, ждёт history, скачивает PNG через /view в test_results/<scene>.png."""
    summary = graph_summary(graph)
    result = SceneResult(
        scene_id=scene.scene_id, variant=scene.variant, hero=scene.hero, seed=TESTWF_SEED, prompt=scene.prompt,
        refs=",".join(f"{k}={v}" for k, v in summary["refs"]), latent_from=str(summary["latent_from"]),
        has_vae=summary["has_vae"])
    started = clock()
    artifact.write_text(f"test_results/graphs/{scene.scene_id}.json",
                        json.dumps(graph, ensure_ascii=False, indent=2))
    try:
        resp = client.post("/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
        resp.raise_for_status()
        prompt_id = str(resp.json()["prompt_id"])
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        result.status = "error"
        result.detail = mask_text(f"очередь ComfyUI: {type(exc).__name__}")
        result.duration_sec = clock() - started
        return result

    deadline = started + SCENE_TIMEOUT_SEC
    images: list[dict] = []
    while True:
        if stop is not None and stop.is_set():
            result.status = "error"
            result.detail = "остановлено сигналом"
            break
        if clock() >= deadline:
            result.status = "timeout"
            result.detail = f"нет результата за {int(SCENE_TIMEOUT_SEC)} с"
            break
        try:
            hist = client.get(f"/history/{prompt_id}").json().get(prompt_id)
        except (httpx.HTTPError, ValueError):
            hist = None  # временный сбой связи: повтор по таймеру, сценарий не отменяем
        if hist:
            status = hist.get("status", {}) or {}
            if status.get("status_str") == "error":
                result.status = "error"
                result.detail = _error_text(status)
                break
            images = _collect_images(hist.get("outputs", {}) or {})
            if images:
                break
        sleep(SCENE_POLL_SEC)

    if result.status in ("error", "timeout") or not images:
        if result.status == "not_run":
            result.status = "error"
            result.detail = "в history нет изображения"
        result.duration_sec = clock() - started
        return result

    first = images[0]
    try:
        got = client.get("/view", params={"filename": first["filename"],
                                          "subfolder": first.get("subfolder", ""),
                                          "type": first.get("type", "output")})
        got.raise_for_status()
    except httpx.HTTPError as exc:
        result.status = "error"
        result.detail = mask_text(f"/view: {type(exc).__name__}")
        result.duration_sec = clock() - started
        return result
    rel = f"test_results/{scene.scene_id}.png"
    artifact.write_bytes(rel, got.content)
    result.png = f"{scene.scene_id}.png"
    result.status = "ok"
    result.duration_sec = clock() - started
    return result


# --- Основной прогон ----------------------------------------------------------------------------

def _write_summary(artifact: _Artifact, summary: ABRunSummary, *, keepalive_pings: int, keepalive_failures: int,
                   finished_note: str) -> None:
    lines = [
        "# Тест сценариев workflow (cartoon_photos, A/B)",
        "",
        f"- Метка карты: `{summary.label}`",
        f"- Номер карты Vast: {summary.instance_id}",
        f"- Готово сценариев: {summary.ok_count} из {len(SCENE_CATALOG)}",
        f"- Keepalive /health: пингов {keepalive_pings}, сбоев {keepalive_failures}",
        f"- Итог: {finished_note}",
        "",
        "| Сцена | Вариант | Герой | Статус | Секунд | PNG | Опорные | Латент | VAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.results:
        lines.append(f"| {r.scene_id} | {VARIANT_NAMES[r.variant]} | {'да' if r.hero else 'нет'} | {r.status} | "
                     f"{r.duration_sec:.1f} | {r.png or '-'} | {r.refs or '-'} | {r.latent_from or '-'} | "
                     f"{'да' if r.has_vae else 'нет'} |")
    artifact.write_text("run/summary.md", "\n".join(lines) + "\n")


def _masked_instance_json(info) -> dict:
    """Срез ответа Vast для instance.json (как RunArtifact.card_instance в orchestrator).

    remove_instance_fields убирает адрес, внешние порты, ssh, extra_env и instance_api_key (в имени
    ключа есть «port», поэтому и поле ports тоже уходит). Затем поле ports добавляется заново, но
    только с внутренними портами 8000 и 18188 и значениями «***» (_internal_ports_view). Текст
    дополнительно маскируется.
    """
    safe = remove_instance_fields(info.raw)
    view = _internal_ports_view(info.raw)
    if view is not None:
        safe["ports"] = view
    return json.loads(mask_text(json.dumps(safe, ensure_ascii=False, default=str)))


def _slot_rel(gen: int) -> str:
    """Каталог артефакта одной карты: cards/slot1_generation<поколение>."""
    return f"cards/slot1_generation{gen}"


def _mask_card_text(text: str, card: "_Card") -> str:
    """Текст, полученный с карты (журнал загрузки, логи): внутренние порты сохраняются, адреса карты маскируются."""
    return mask_external(mask_fetched_text(text, keep_internal_ports=True), card.external_values())


class _CardFailed(Exception):
    """Карта не годна к работе: её нужно снять и взять следующее предложение. Текст причины фиксирован."""


@dataclass
class _Card:
    """Одна арендованная карта в этом запуске. Адреса живут только в памяти и в артефакте маскируются."""
    instance_id: int
    offer_id: int
    machine_id: Optional[int]
    price: float
    gen: int
    rented_at: float
    slot_rel: str
    url: Optional[str] = None
    public_ip: Optional[str] = None
    direct_port: Optional[int] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    misses: int = 0           # подряд ошибок опроса Vast; сбрасывается при любом успешном ответе
    rate_set: bool = False    # ставка из ответа Vast уже передана в учёт бюджета

    def external_values(self) -> list:
        return [self.public_ip, self.direct_port, self.ssh_host, self.ssh_port]


@dataclass
class _RentState:
    """Состояние аренды в пределах одного запуска (SPEC 8.2)."""
    bad_machines: set = field(default_factory=set)   # постоянный список bad_hosts.txt и машины снятых карт
    bad_offers: set = field(default_factory=set)     # предложения, по которым аренда не удалась
    used_offers: set = field(default_factory=set)    # предложения, уже взятые в аренду в этом запуске
    rent_requests: int = 0                           # запросы аренды (с ошибкой и без): лимит max_rent_tries
    generation: int = 0                              # поколение последней карты
    current: Optional["_Card"] = None                # карта, которая сейчас учитывается (уборка в finally)


def _rental_price(offer, cfg) -> float:
    """Цена для бюджета и заявки аренды.

    Прерываемая аренда: ставка bid = цена предложения + bid_margin, округлённая до 3 знаков
    (как в оркестраторе и vast_stable_run.sh). Обычная: dph_total, а если его нет, цена предложения.
    """
    if cfg.interruptible:
        return round(float(offer.price_per_hr) + float(cfg.bid_margin), 3)
    return offer.dph_total if offer.dph_total > 0 else offer.price_per_hr


def _outbid(info) -> bool:
    """Прерываемая карта перебита: по статусу или по тексту статуса Vast (маркер «outbid»)."""
    return info.actual_status == "outbid" or "outbid" in (info.status_msg or "").lower()


def _rent_next(vast, cfg, state: _RentState, budget, *, image: str, label: str, onstart: str,
               env: dict) -> _Card:
    """Арендует первое подходящее предложение. Ошибка создания — следующее предложение; исчерпание — ABRuntimeError."""
    if state.rent_requests >= cfg.max_rent_tries:
        raise ABRuntimeError(f"исчерпан лимит запросов аренды (max_rent_tries = {cfg.max_rent_tries})")
    try:
        offers = vast.search_offers(_offer_filter(cfg))
    except VastError as exc:
        raise ABRuntimeError(f"поиск предложений Vast не удался: {type(exc).__name__}") from exc
    eligible = [o for o in offers
                if o.offer_id not in state.bad_offers and o.offer_id not in state.used_offers
                and o.machine_id not in state.bad_machines]
    eligible.sort(key=lambda o: o.price_per_hr)
    for offer in eligible:
        price = _rental_price(offer, cfg)
        if cfg.interruptible and price > cfg.max_price_per_hour:
            log.warning("Предложение %d пропущено: ставка bid %.3f USD/ч выше лимита %.2f USD/ч.",
                        offer.offer_id, price, cfg.max_price_per_hour)
            continue
        if not budget.can_afford(price, AFFORD_ESTIMATE_SEC):
            log.warning("Бюджета не хватает на предложение %d (остаток %.4f USD).",
                        offer.offer_id, budget.remaining_budget())
            continue
        if state.rent_requests >= cfg.max_rent_tries:
            raise ABRuntimeError(f"исчерпан лимит запросов аренды (max_rent_tries = {cfg.max_rent_tries})")
        state.rent_requests += 1
        state.used_offers.add(offer.offer_id)
        try:
            instance_id = vast.create_instance(
                offer.offer_id, image, cfg.disk_gb, env_vars=env, label=label, onstart=onstart,
                bid_price=price if cfg.interruptible else None)
        except VastError as exc:
            state.bad_offers.add(offer.offer_id)
            log.warning("Не удалось создать карту по предложению %d: %s. Пробую следующее предложение.",
                        offer.offer_id, type(exc).__name__)
            continue
        budget.track_instance_start(instance_id, price)
        state.generation += 1
        log.info("Арендована карта %d по предложению %d: %.4f USD/ч (поколение %d).",
                 instance_id, offer.offer_id, price, state.generation)
        return _Card(instance_id=instance_id, offer_id=offer.offer_id, machine_id=offer.machine_id,
                     price=price, gen=state.generation, rented_at=time.monotonic(),
                     slot_rel=_slot_rel(state.generation))
    raise ABRuntimeError("нет предложения, которое проходит фильтр цены, бюджета и списка плохих машин")


def _wait_ready(vast, card: _Card, cfg, token: str, budget, stop: Optional[threading.Event]) -> None:
    """Опрос карты до готовности: /health ok, ComfyUI отвечает, есть SSH-адрес.

    Возвращается при готовности. Причина замены карты — _CardFailed. Остановка сигналом и правило A
    (ComfyUI не ответил за 5 мин) — ABRuntimeError, как раньше.
    """
    silent_sec = cfg.silent_host_timeout_min * 60
    while True:
        if stop is not None and stop.is_set():
            raise ABRuntimeError("остановлено сигналом")
        elapsed = time.monotonic() - card.rented_at
        if elapsed >= silent_sec:
            raise _CardFailed(f"карта не ответила за {cfg.silent_host_timeout_min} мин")
        try:
            info = vast.get_instance(card.instance_id)
        except VastError as exc:
            # VastInstanceNotFound тоже сюда; оба вида ошибок считаются подряд до NOT_FOUND_LIMIT
            card.misses += 1
            if isinstance(exc, VastInstanceNotFound):
                log.warning("Карта %d не найдена в ответе Vast (подряд: %d из %d).",
                            card.instance_id, card.misses, NOT_FOUND_LIMIT)
                reason = "экземпляр не найден в Vast"
            else:
                log.warning("Опрос карты %d: ошибка Vast %s (подряд: %d из %d).",
                            card.instance_id, type(exc).__name__, card.misses, NOT_FOUND_LIMIT)
                reason = "Vast не отвечает по карте"
            if card.misses >= NOT_FOUND_LIMIT:
                raise _CardFailed(reason)
            time.sleep(POLL_SEC)
            continue
        card.misses = 0  # любой успешный ответ обнуляет счётчик
        card.public_ip = info.public_ip or card.public_ip
        card.direct_port = info.direct_port or card.direct_port
        card.ssh_host = info.ssh_host or card.ssh_host
        card.ssh_port = info.ssh_port or card.ssh_port
        if info.dph > 0 and not card.rate_set:
            card.rate_set = True
            try:
                budget.set_dph(card.instance_id, info.dph)
            except BudgetError as exc:
                log.warning("Ставка карты %d в учёте бюджета не обновлена: %s.", card.instance_id, type(exc).__name__)
        if info.actual_status in STOPPED_AFTER_START_STATUSES or info.intended_status == "stopped":
            raise _CardFailed("карта остановлена хостом")
        if info.status_msg and _has_start_marker(info.status_msg):
            raise _CardFailed("ошибка запуска карты по тексту статуса Vast")
        if cfg.interruptible and _outbid(info):
            raise _CardFailed("прерываемая карта перебита другой ставкой")
        if info.disk_gb is not None and info.disk_gb < cfg.disk_gb:
            raise _CardFailed("диск карты меньше заданного")
        if info.actual_status == "running" and info.public_ip and info.direct_port:
            card.url = f"http://{info.public_ip}:{info.direct_port}"
            health = _probe_health(card.url, token)
            if health is not None:
                if health.get("comfy_state") == "dead":
                    raise _CardFailed("процесс ComfyUI на карте завершился")
                if health.get("reason") == "weights_failed":
                    raise _CardFailed("воркер сообщил о сбое загрузки весов")
                if health.get("ok") is True:
                    if health.get("comfy_alive") is True and card.ssh_host and card.ssh_port:
                        return
                    if "comfy_alive" in health and health.get("comfy_alive") is not True and \
                            elapsed >= COMFY_ANSWER_LIMIT_SEC:
                        raise ABRuntimeError("ComfyUI на карте не ответил за 5 мин от аренды (правило A)")
        time.sleep(POLL_SEC)


def _save_card_logs(vast, card: _Card, token: str, artifact: "_Artifact") -> None:
    """Журнал снимаемой карты: GET /logs воркера, если адрес известен, иначе логи контейнера через Vast.

    Текст маскируется (mask_fetched_text + адреса карты). Адрес результата Vast не логируется.
    """
    rel = f"{card.slot_rel}/logs.txt"
    if card.url:
        try:
            resp = httpx.get(card.url + "/logs", headers={"X-Worker-Token": token}, timeout=HTTP_TIMEOUT_SEC)
            if resp.status_code == 200:
                artifact.write_text(rel, _mask_card_text(resp.text, card))
                log.info("Журнал воркера снятой карты сохранён.")
                return
            log.warning("GET /logs воркера: HTTP %d.", resp.status_code)
        except httpx.HTTPError as exc:
            log.warning("GET /logs воркера не удался: %s.", type(exc).__name__)
    try:
        result_url = vast.request_logs(card.instance_id, tail=LOGS_TAIL_LINES)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "vast_logs.txt")
            code = vast.download_logs(result_url, dest)
            if code == 200 and os.path.exists(dest):
                text = Path(dest).read_text(encoding="utf-8", errors="replace")
                artifact.write_text(rel, _mask_card_text(text, card))
                log.info("Логи контейнера снятой карты через Vast сохранены.")
            else:
                log.warning("Логи контейнера через Vast не загружены: HTTP %d.", code)
    except (VastError, OSError) as exc:
        log.warning("Логи контейнера через Vast не получены: %s.", type(exc).__name__)


def _retire_card(vast, card: _Card, reason: str, *, budget, token: str, artifact: "_Artifact",
                 state: _RentState) -> None:
    """Снимает плохую карту: логи (best effort), учёт бюджета, уничтожение. Ошибки уборки не прерывают запуск."""
    log.warning("Карта %d снимается: %s.", card.instance_id, reason)
    artifact.write_text(f"{card.slot_rel}/status.txt", f"снята: {reason}\n")
    if card.machine_id is not None:
        state.bad_machines.add(card.machine_id)  # машина больше не предлагается в этом запуске
    try:
        _save_card_logs(vast, card, token, artifact)
    except Exception as exc:  # noqa: BLE001 - логи не должны срывать замену карты
        log.warning("Логи снимаемой карты не сохранены: %s.", type(exc).__name__)
    try:
        budget.track_instance_stop(card.instance_id)
    except BudgetError as exc:
        log.warning("Учёт бюджета карты %d при снятии: %s.", card.instance_id, type(exc).__name__)
    try:
        vast.destroy_instance(card.instance_id)
    except VastError as exc:
        log.error("Снятие карты %d не подтверждено: %s. Проверьте kill-cards.", card.instance_id, type(exc).__name__)
    state.current = None


def _acquire_ready_card(vast, cfg, state: _RentState, budget, artifact: "_Artifact", *, token: str, image: str,
                        label: str, onstart: str, env: dict, meta_base: dict, summary: "ABRunSummary",
                        stop: Optional[threading.Event]) -> _Card:
    """Аренда и ожидание до готовности; плохие карты снимаются, берётся следующее предложение."""
    while True:
        card = _rent_next(vast, cfg, state, budget, image=image, label=label, onstart=onstart, env=env)
        state.current = card
        summary.instance_id = card.instance_id
        artifact.write_text(f"{card.slot_rel}/meta.json", json.dumps({
            **meta_base, "instance_id": card.instance_id, "offer_id": card.offer_id,
            "price_per_hr": card.price, "generation": card.gen,
        }, ensure_ascii=False, indent=2) + "\n")
        artifact.write_text(f"{card.slot_rel}/status.txt", "rented\n")
        try:
            _wait_ready(vast, card, cfg, token, budget, stop)
        except _CardFailed as exc:
            _retire_card(vast, card, str(exc), budget=budget, token=token, artifact=artifact, state=state)
            continue
        artifact.write_text(f"{card.slot_rel}/status.txt", "ready\n")
        return card


def run_ab_test(vast, cfg, *, run_id: str = "", worker_build: str = "", repo: str = "",
                artifact_dir: str = "", docker_image: Optional[str] = None,
                stop: Optional[threading.Event] = None) -> ABRunSummary:
    """Один прогон: аренда карты, 16 сцен через SSH-туннель, PNG в артефакт, уборка в finally.

    Ошибки входа (номер сборки, номер запуска, сборка графов, bad_hosts.txt) бросаются до аренды: ABInputError.
    Ошибки аренды, карты, туннеля, ComfyUI бросаются как ABRuntimeError. Плохая карта (не запустилась,
    мёртвый ComfyUI, перебита, нет диска и т.п.) снимается, и берётся следующее предложение в пределах
    max_rent_tries и бюджета. Карта в любом случае уничтожается; при сбое уничтожения срабатывает
    destroy_by_label. Релизы (Release) не затрагиваются.
    """
    # --- проверка входа до любых трат ---
    if not _valid_worker_build(worker_build):
        raise ABInputError("Номер сборки воркера (--worker-build) должен быть из 7-64 символов 0-9 и a-f.")
    run_id = (run_id or "").strip() or str(int(time.time()))
    if not re.fullmatch(r"[0-9]+", run_id):
        raise ABInputError("Номер запуска (--run-id) должен состоять из цифр.")
    image = (docker_image or "").strip() or TESTWF_IMAGE
    try:
        onstart = _worker_onstart(worker_build)
    except gp.GeminiInputError as exc:
        raise ABInputError(str(exc)) from exc
    try:
        bad_hosts = load_bad_hosts()  # постоянный список плохих машин; ошибка формата — код 2 до аренды
    except BadHostsError as exc:
        raise ABInputError(str(exc)) from exc

    template, ref_nodes = load_template()
    graphs: dict[str, dict] = {}
    for scene in SCENE_CATALOG:
        graphs[scene.scene_id] = build_graph(template, ref_nodes, scene)  # GraphError = ABInputError

    label = f"testwf-{run_id}"
    artifact = _Artifact(artifact_dir or None)
    summary = ABRunSummary(label=label)
    for scene in SCENE_CATALOG:
        summary.results.append(SceneResult(scene.scene_id, scene.variant, scene.hero, TESTWF_SEED, scene.prompt))
    artifact.write_text("test_results/index.txt", INDEX_HEADER)

    token = secrets.token_urlsafe(24)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    # ключ SSH маскируется и целиком, и построчно (как в generate.yml через secret_mask_values)
    set_known_secrets([token, hf_token, *secret_mask_values(os.environ.get("VAST_SSH_KEY", ""))])
    env = {
        "WORKER_TOKEN": token,
        "WORKER_BUILD": worker_build,  # без него bootstrap.sh завершается с кодом 2 на шаге 1
        "WORKER_IDLE_TIMEOUT_MIN": str(cfg.worker_idle_timeout_min),
        "MODEL_PROFILE": MODEL_PROFILE,
        "COMFYUI_ARGS": COMFYUI_ARGS,
        "GEN_LOG_LEVEL": os.environ.get("GEN_LOG_LEVEL", "INFO"),
        # без проброса порта Vast не открывает 8000 наружу (как в orchestrator.py)
        f"-p {WORKER_HTTP_PORT}:{WORKER_HTTP_PORT}": "1",
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token  # пробрасывается как у генерации; в журнал не пишется
    meta_base = {
        "label": label, "worker_build": worker_build, "repo": repo, "run_id": run_id,
        "image": image, "scenes": len(SCENE_CATALOG), "seed": TESTWF_SEED,
    }

    budget = BudgetTracker(cfg.budget_limit_usd)
    state = _RentState(bad_machines=set(bad_hosts))
    keepalive: Optional[_Keepalive] = None
    status_note = "не начато"
    keepalive_pings = keepalive_failures = 0
    try:
        # --- аренда и ожидание готовности (с заменой плохих карт) ---
        card = _acquire_ready_card(vast, cfg, state, budget, artifact, token=token, image=image, label=label,
                                   onstart=onstart, env=env, meta_base=meta_base, summary=summary, stop=stop)
        log.info("Карта готова: /health ok, ComfyUI отвечает. Открываю SSH-туннель к порту %d.", COMFY_REMOTE_PORT)

        # --- keepalive до конца сцен ---
        keepalive = _Keepalive(card.url, token)
        keepalive.start()

        key = os.environ.get("VAST_SSH_KEY", "")
        try:
            with ssh_probe.open_tunnel(card.ssh_host, card.ssh_port, key, COMFY_REMOTE_PORT,
                                       ready_timeout_sec=TUNNEL_READY_SEC) as local_port:
                base = f"http://127.0.0.1:{local_port}"
                with httpx.Client(base_url=base, timeout=HTTP_TIMEOUT_SEC) as client:
                    for index, scene in enumerate(SCENE_CATALOG):
                        if stop is not None and stop.is_set():
                            raise ABRuntimeError("остановлено сигналом")
                        log.info("Сцена %d/%d: %s (%s).", index + 1, len(SCENE_CATALOG), scene.scene_id,
                                 VARIANT_NAMES[scene.variant])
                        result = _render_scene(client, scene, graphs[scene.scene_id], artifact, stop)
                        summary.results[index] = result
                        artifact.append_text("test_results/index.txt", result.index_line())
                        log.info("Сцена %s: статус %s, %.1f с.", scene.scene_id, result.status, result.duration_sec)
        except ssh_probe.TunnelError as exc:
            raise ABRuntimeError(f"SSH-туннель не открылся: {exc.failure_class}") from exc
        status_note = "все сцены обработаны"
    except ABRuntimeError as exc:
        status_note = f"прервано: {exc}"
        log.error("Тест прерван: %s", exc)
        raise
    finally:
        if keepalive is not None:
            keepalive.stop()
            keepalive_pings, keepalive_failures = keepalive.pings, keepalive.failures
        # --- уборка: журнал карты, instance.json (маскированный), учёт бюджета, уничтожение по метке ---
        cur = state.current  # карта, которая ещё учитывается (снятые карты сюда уже не попадают)
        if cur is not None and cur.ssh_host and cur.ssh_port:
            try:
                res = ssh_probe.run_probe(cur.ssh_host, cur.ssh_port, os.environ.get("VAST_SSH_KEY", ""),
                                          "bootstrap_tail")
                if res.ok:
                    artifact.write_text("run/worker_bootstrap_tail.txt", _mask_card_text(res.output, cur))
                else:
                    artifact.write_text("run/worker_bootstrap_tail.txt", f"не получен: {res.failure_class}\n")
            except Exception as exc:  # noqa: BLE001 - журнал карты не должен срывать уборку
                log.warning("Журнал карты не снят: %s.", type(exc).__name__)
        if cur is not None:
            try:
                info = vast.get_instance(cur.instance_id)
                artifact.write_text("run/instance.json",
                                    json.dumps(_masked_instance_json(info), ensure_ascii=False, indent=2) + "\n")
            except VastError as exc:
                log.warning("instance.json не записан: %s.", type(exc).__name__)
            try:
                budget.track_instance_stop(cur.instance_id)
            except BudgetError as exc:  # учёт не должен мешать уничтожению карты
                log.warning("Учёт бюджета при снятии карты: %s.", type(exc).__name__)
        try:
            destroyed = vast.destroy_by_label(label)
            log.info("Уничтожено карт по метке %s: %d.", label, len(destroyed))
        except VastError as exc:
            log.error("Уничтожение по метке %s не удалось: %s. Проверьте kill-cards.", label, type(exc).__name__)
        _write_summary(artifact, summary, keepalive_pings=keepalive_pings,
                       keepalive_failures=keepalive_failures, finished_note=status_note)
        artifact.write_text(f"{_slot_rel(max(state.generation, 1))}/status.txt", "finished\n")

    return summary
