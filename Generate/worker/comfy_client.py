"""Клиент локального headless ComfyUI (SPEC разделы 5, 8.1, 8.4).

Только стандартная библиотека и httpx (SPEC 0.2). Все сообщения на русском (SPEC 0.3).
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, Optional

import httpx

log = logging.getLogger("Generate.worker.comfy_client")

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT = 600.0
REQUIRED_BINDINGS = ("prompt", "num_frames", "seed", "filename_prefix")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".gif")
OUTPUT_KINDS = ("videos", "gifs", "images")


# ---------------------------------------------------------------------------
# Исключения
# ---------------------------------------------------------------------------

class ComfyError(Exception):
    """Базовая ошибка клиента ComfyUI (текст на русском)."""


class ComfyConnectionError(ComfyError):
    """ComfyUI недоступен или соединение не удалось."""


class ComfyExecutionError(ComfyError):
    """ComfyUI вернул ошибку выполнения."""


class ComfyTimeoutError(ComfyError):
    """Выполнение превысило допустимое время."""


# ---------------------------------------------------------------------------
# Подстановка параметров в workflow (SPEC раздел 5)
# ---------------------------------------------------------------------------

def _set_input(graph: dict, binding: Any, name: str, value: Any) -> None:
    if not isinstance(binding, dict) or "node_id" not in binding or "field" not in binding:
        raise ComfyError(f"Привязка «{name}» должна содержать ключи node_id и field.")
    node_id = str(binding["node_id"])
    field = str(binding["field"])
    node = graph.get(node_id)
    if not isinstance(node, dict):
        raise ComfyError(f"В workflow нет узла «{node_id}» (привязка «{name}»).")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ComfyError(f"У узла «{node_id}» нет раздела inputs (привязка «{name}»).")
    inputs[field] = value


def inject_workflow(workflow_graph: dict, bindings: dict, prompt: str, num_frames: int,
                    seed: int, filename_prefix: str,
                    extra_inputs: Optional[dict] = None) -> dict:
    """Возвращает глубокую копию workflow с подставленными значениями.

    bindings: {имя: {"node_id": "6", "field": "text"}}. Обязательные имена:
    prompt, num_frames, seed, filename_prefix. extra_inputs: {имя_привязки: значение};
    каждое имя должно присутствовать в bindings.
    """
    missing = [k for k in REQUIRED_BINDINGS if k not in bindings]
    if missing:
        raise ComfyError("В привязках не хватает обязательных полей: " + ", ".join(missing) + ".")
    graph = copy.deepcopy(workflow_graph)
    values = {"prompt": prompt, "num_frames": int(num_frames), "seed": int(seed),
              "filename_prefix": filename_prefix}
    for name, value in values.items():
        _set_input(graph, bindings[name], name, value)
    for name, value in (extra_inputs or {}).items():
        if name not in bindings:
            raise ComfyError(f"Для дополнительного входа «{name}» нет привязки.")
        _set_input(graph, bindings[name], name, value)
    return graph


# ---------------------------------------------------------------------------
# Разбор истории
# ---------------------------------------------------------------------------

def _error_text(entry: dict) -> str:
    status = entry.get("status") or {}
    for item in status.get("messages") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "execution_error":
            data = item[1] if isinstance(item[1], dict) else {}
            msg = data.get("exception_message") or "без пояснения"
            node = data.get("node_id")
            return f"{msg} (узел {node})" if node else str(msg)
    return "причина не указана"


def extract_output_path(entry: dict) -> str:
    """Путь («подпапка/имя» или «имя») первого видеофайла из outputs записи истории."""
    outputs = entry.get("outputs") or {}
    for node_id in sorted(outputs, key=str):
        node_out = outputs[node_id]
        if not isinstance(node_out, dict):
            continue
        for kind in OUTPUT_KINDS:
            for item in node_out.get(kind) or []:
                name = item.get("filename") if isinstance(item, dict) else None
                if name and name.lower().endswith(VIDEO_EXTENSIONS):
                    sub = item.get("subfolder") or ""
                    return f"{sub}/{name}" if sub else name
    raise ComfyExecutionError("ComfyUI завершил задачу, но видеофайл в результатах не найден.")


# ---------------------------------------------------------------------------
# Клиент
# ---------------------------------------------------------------------------

class ComfyClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT,
                 client: Optional[httpx.Client] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=30.0)

    # --- служебное ---
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            raise ComfyConnectionError(
                f"ComfyUI недоступен по адресу {self.base_url}: {type(exc).__name__}.") from exc

    @staticmethod
    def _json(resp: httpx.Response, what: str) -> Any:
        try:
            return resp.json()
        except ValueError as exc:
            raise ComfyError(f"ComfyUI вернул некорректный JSON ({what}).") from exc

    # --- API ---
    def check_health(self) -> dict:
        resp = self._request("GET", "/system_stats")
        if resp.status_code != 200:
            raise ComfyConnectionError(
                f"ComfyUI ответил на проверку состояния кодом {resp.status_code}.")
        data = self._json(resp, "system_stats")
        if not isinstance(data, dict):
            raise ComfyError("Ответ system_stats имеет неожиданный формат.")
        return data

    def queue_prompt(self, prompt_workflow: dict) -> str:
        resp = self._request("POST", "/prompt", json={"prompt": prompt_workflow})
        if resp.status_code != 200:
            detail = ""
            try:
                body = resp.json()
                err = body.get("error") if isinstance(body, dict) else None
                if isinstance(err, dict):
                    detail = str(err.get("message") or "")
                elif err:
                    detail = str(err)
            except ValueError:
                pass
            raise ComfyExecutionError(
                f"ComfyUI отклонил задачу (код {resp.status_code})" + (f": {detail}" if detail else "."))
        data = self._json(resp, "prompt")
        prompt_id = data.get("prompt_id") if isinstance(data, dict) else None
        if not prompt_id:
            raise ComfyError("В ответе ComfyUI нет идентификатора задачи prompt_id.")
        log.info("Задача поставлена в очередь ComfyUI: %s", prompt_id)
        return str(prompt_id)

    def poll_progress(self, prompt_id: str) -> dict:
        """Состояние задачи: {"state": "pending"|"success"|"error", "entry": dict|None}."""
        resp = self._request("GET", f"/history/{prompt_id}")
        if resp.status_code != 200:
            raise ComfyConnectionError(
                f"ComfyUI ответил на запрос истории кодом {resp.status_code}.")
        data = self._json(resp, "history")
        entry = data.get(prompt_id) if isinstance(data, dict) else None
        if not isinstance(entry, dict):
            return {"state": "pending", "entry": None}
        status = entry.get("status") or {}
        status_str = status.get("status_str")
        if status_str == "error":
            return {"state": "error", "entry": entry}
        if status_str == "success" or status.get("completed") or entry.get("outputs"):
            return {"state": "success", "entry": entry}
        return {"state": "pending", "entry": entry}

    def wait_for_completion(self, prompt_id: str, timeout: float = DEFAULT_TIMEOUT,
                            poll_interval: float = 1.0) -> str:
        start = time.monotonic()
        log.info("[%s] Этап: постановка в очередь, задача %s.", time.strftime("%H:%M:%S"), prompt_id)
        announced = False
        while True:
            state = self.poll_progress(prompt_id)
            if state["state"] == "error":
                raise ComfyExecutionError(
                    f"Ошибка выполнения в ComfyUI: {_error_text(state['entry'])}.")
            if state["state"] == "success":
                path = extract_output_path(state["entry"])
                log.info("[%s] Этап: завершено, результат %s (%.1f с).",
                         time.strftime("%H:%M:%S"), path, time.monotonic() - start)
                return path
            if not announced:
                log.info("[%s] Этап: сэмплирование и выполнение графа.", time.strftime("%H:%M:%S"))
                announced = True
            if time.monotonic() - start >= timeout:
                raise ComfyTimeoutError(
                    f"Задача {prompt_id} не завершилась за {timeout:g} с.")
            time.sleep(poll_interval)

    # --- жизненный цикл ---
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ComfyClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
