"""Ручной замер: выдерживает ли один ключ OpenRouter три параллельных запроса.

Один прогон, без сравнения и без базового одиночного запроса. Скользящее окно из 3 запросов
над 6 батчами по 25 клипов. Сегменты: первые 150 из корневого final.srt, батчи строит публичный
build_requests без изменений. Повторов нет. Ключ берётся только из OPENROUTER10.
Результаты пишутся после каждого запроса в BENCH_OUT_DIR.

Запуск из корня репозитория:
  OPENROUTER10=... BENCH_OUT_DIR=... python Generate/tools/bench_openrouter_parallel.py
Необязательные переменные: MODEL (по умолчанию nvidia/nemotron-3.5-super-vl:free),
TIMEOUT_SEC (по умолчанию OPENROUTER_TIMEOUT_SEC транспорта, 600 с).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# корень репозитория: нужен для импорта пакета Generate и для относительного пути стиля профиля
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from Generate.cli import _build_prompt_config, _load_cfg, _load_srt  # noqa: E402
from Generate.core.gemini_prompts import GeminiInputError, build_requests  # noqa: E402
from Generate.core.openrouter_transport import OPENROUTER_TIMEOUT_SEC  # noqa: E402

KEY_VAR = "OPENROUTER10"                       # единственный ключ этого замера
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
SRT_NAME = "final.srt"                         # корневой файл субтитров
PROFILE = "ltx25"                              # те же значения, что у CLI для этого профиля
BATCH_SIZE = 25
PRESET = "none"
TOTAL_SEGMENTS = 150                           # первые сегменты SRT
EXPECTED_BATCHES = 6
WINDOW = 3                                     # размер скользящего окна (max_workers)
ERROR_PREVIEW = 200                            # сколько символов текста ошибки попадает в записи
JSONL_NAME = "bench_openrouter_parallel.jsonl"
JSON_NAME = "bench_openrouter_parallel.json"
TXT_NAME = "bench_openrouter_parallel.txt"

# шаблоны похожих на ключ фрагментов: как в транспорте (sk-..., AIza..., заголовок Bearer)
_SECRET_PATTERNS = (
    re.compile(r"sk-[0-9A-Za-z_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{8,}"),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
)

# httpx пишет адреса запросов на уровне INFO; для замера это лишний шум
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class Masker:
    """Заменяет значение ключа и похожие на ключ фрагменты на *** в каждой строке."""

    def __init__(self, key: str):
        self._key = key

    def text(self, value) -> str:
        out = str(value)
        if self._key:
            out = out.replace(self._key, "***")
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub("***", out)
        return out

    def obj(self, value):
        # маскируются строки внутри структур, а не готовая JSON-строка: так JSON не ломается
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(k): self.obj(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.obj(v) for v in value]
        return value


class Sink:
    """Вывод в консоль и файлы в BENCH_OUT_DIR. Запись строк jsonl идёт под одной блокировкой."""

    def __init__(self, out_dir: Path, masker: Masker):
        self.out_dir = out_dir
        self.masker = masker
        self._lock = threading.Lock()

    def say(self, text: str = "") -> None:
        line = self.masker.text(text)
        with self._lock:
            print(line, flush=True)

    def append_record(self, record: dict) -> None:
        # файл открывается на дозапись, строка пишется, сбрасывается и закрывается сразу
        line = json.dumps(self.masker.obj(record), ensure_ascii=False)
        with self._lock:
            with open(self.out_dir / JSONL_NAME, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()


@dataclass
class RunContext:
    """Общие для всех запросов значения прогона. Заголовки с ключом сюда попадают, но не выводятся."""
    client: httpx.Client
    url: str
    headers: dict
    model: str
    max_tokens: int
    effort: str
    t0: float
    sink: Sink
    masker: Masker


def build_body(model: str, batch, max_tokens: int, effort: str) -> dict:
    """Тело запроса: поля и порядок те же, что в OpenRouterTransport.generate_json."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": batch.system}, {"role": "user", "content": batch.user}],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "segments", "schema": batch.schema}},
    }
    # reasoning добавляется только при уровне не none, как в транспорте
    if effort != "none":
        body["reasoning"] = {"effort": effort}
    return body


def fill_from_response(record: dict, resp, masker: Masker) -> None:
    """Заполняет поля записи из ответа. Заголовки берутся только x-ratelimit-* и retry-after."""
    headers = resp.headers
    retry = headers.get("retry-after")
    record["retry_after"] = masker.text(retry) if retry else None
    record["ratelimit"] = {k: masker.text(v) for k, v in headers.items() if k.lower().startswith("x-ratelimit")}
    try:
        text = resp.text
    except Exception:
        text = ""
    if resp.status_code != 200:
        record["error"] = masker.text(text or "тело ответа пусто")[:ERROR_PREVIEW]
        return
    try:
        data = resp.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        record["error"] = "ответ 200 не разобран: нет choices"
        return
    content = message.get("content")
    reasoning = message.get("reasoning") if "reasoning" in message else message.get("reasoning_content")
    usage = data.get("usage")
    record["finish_reason"] = choice.get("finish_reason")
    record["usage"] = usage
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["completion_tokens"] = usage.get("completion_tokens")
        record["reasoning_tokens"] = details.get("reasoning_tokens") if isinstance(details, dict) else None
    record["content_len"] = len(content) if isinstance(content, str) else None
    record["reasoning_len"] = len(reasoning) if isinstance(reasoning, str) else None
    record["empty_content"] = not (isinstance(content, str) and content.strip())


def progress_line(r: dict) -> str:
    status = r["http_status"] if r["http_status"] is not None else "нет ответа"
    return (f"Батч {r['batch']} (клипы {r['clip_range']}): старт {r['start_s']} с, конец {r['end_s']} с, "
            f"HTTP {status}, finish_reason {r['finish_reason']}, completion {r['completion_tokens']}, "
            f"reasoning {r['reasoning_tokens']}, content {r['content_len']} симв.")


def send_batch(index: int, batch, ctx: RunContext) -> dict:
    """Один запрос без повторов. Запись в файл сразу после ответа или сбоя."""
    nums = batch.nums
    record = {
        "batch": index, "clip_range": f"{nums[0]}-{nums[-1]}", "clips": len(nums),
        "start_s": None, "end_s": None, "duration_s": None,
        "http_status": None, "error": None, "finish_reason": None,
        "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None, "usage": None,
        "content_len": None, "reasoning_len": None, "empty_content": False,
        "retry_after": None, "ratelimit": {},
    }
    sent = time.monotonic()
    record["start_s"] = round(sent - ctx.t0, 1)
    try:
        body = build_body(ctx.model, batch, ctx.max_tokens, ctx.effort)
        resp = ctx.client.post(ctx.url, headers=ctx.headers, json=body)
        record["http_status"] = resp.status_code
        fill_from_response(record, resp, ctx.masker)
    except Exception as exc:  # сеть, таймаут или непредвиденный сбой: без повтора
        record["error"] = ctx.masker.text(f"{type(exc).__name__}: {exc}")[:ERROR_PREVIEW]
    finished = time.monotonic()
    record["end_s"] = round(finished - ctx.t0, 1)
    record["duration_s"] = round(finished - sent, 1)
    ctx.sink.append_record(record)
    ctx.sink.say(progress_line(record))
    return record


def summarize(records: list, total_s: float) -> dict:
    statuses = [r["http_status"] for r in records]
    return {
        "total_wall_s": round(total_s, 1),
        "requests": [{"batch": r["batch"], "start_s": r["start_s"], "end_s": r["end_s"]} for r in records],
        "count_200": sum(1 for s in statuses if s == 200),
        "count_429": sum(1 for s in statuses if s == 429),
        "count_5xx": sum(1 for s in statuses if isinstance(s, int) and 500 <= s <= 599),
        "count_200_empty": sum(1 for r in records if r["http_status"] == 200 and r["empty_content"]),
        "count_finish_length": sum(1 for r in records if r["finish_reason"] == "length"),
        "count_no_response": sum(1 for s in statuses if s is None),
    }


def summary_text(model: str, summary: dict, records: list) -> str:
    lines = [
        "Параллельный замер OpenRouter: окно 3, 6 батчей по 25 клипов, один прогон",
        f"Модель: {model}",
        f"Общее время прогона: {summary['total_wall_s']} с",
        "",
        "Запросы (старт и конец отсчитываются от начала прогона):",
    ]
    lines += ["  " + progress_line(r) for r in records]
    lines += [
        "",
        "Сводка:",
        f"  ответов 200: {summary['count_200']}",
        f"  429: {summary['count_429']}",
        f"  5xx: {summary['count_5xx']}",
        f"  200 с пустым content: {summary['count_200_empty']}",
        f"  finish_reason length: {summary['count_finish_length']}",
        f"  без ответа (сеть, таймаут): {summary['count_no_response']}",
    ]
    return "\n".join(lines)


def main() -> int:
    # ключ: только OPENROUTER10; при пустом значении печатается имя переменной и выход с кодом 2
    key = os.environ.get(KEY_VAR, "").strip()
    if not key:
        print(f"Ошибка: переменная {KEY_VAR} пуста или не задана. Значение не выводится.", flush=True)
        return 2
    masker = Masker(key)

    out_raw = os.environ.get("BENCH_OUT_DIR", "").strip()
    if not out_raw:
        print("Ошибка: не задана переменная BENCH_OUT_DIR.", flush=True)
        return 2
    out_dir = Path(out_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = Sink(out_dir, masker)

    model = os.environ.get("MODEL", "").strip() or DEFAULT_MODEL
    timeout_raw = os.environ.get("TIMEOUT_SEC", "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else OPENROUTER_TIMEOUT_SEC
    except ValueError:
        sink.say("Ошибка: переменная TIMEOUT_SEC должна быть числом.")
        return 2

    # конфиг и SRT читаются теми же функциями, что у CLI; относительный путь стиля требует корня репозитория
    os.chdir(REPO_ROOT)
    cfg, code = _load_cfg(None)
    if code is not None:
        return code
    segments, code = _load_srt(str(REPO_ROOT / SRT_NAME))
    if code is not None:
        return code
    if len(segments) < TOTAL_SEGMENTS:
        sink.say(f"Ошибка: в {SRT_NAME} меньше {TOTAL_SEGMENTS} сегментов ({len(segments)}).")
        return 2

    # батчи: builder получает ровно первые 150 сегментов и их номера
    head = segments[:TOTAL_SEGMENTS]
    wanted = [s.num for s in head]
    try:
        prompt_cfg = _build_prompt_config(PROFILE, BATCH_SIZE, PRESET)
        batches = build_requests(head, wanted, prompt_cfg)
    except GeminiInputError as exc:
        sink.say(f"Ошибка входных данных: {exc}")
        return 2
    sizes = [len(b.nums) for b in batches]
    if len(batches) != EXPECTED_BATCHES or any(n != BATCH_SIZE for n in sizes):
        sink.say(f"Ошибка: ожидалось {EXPECTED_BATCHES} батчей по {BATCH_SIZE}, получено {sizes}.")
        return 2

    # новый прогон: файл строк очищается, старые строки не смешиваются с новыми
    (out_dir / JSONL_NAME).write_text("", encoding="utf-8")
    url = f"{cfg.openrouter_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sink.say(f"Прогон: модель {model}, окно {WINDOW}, батчей {len(batches)} по {BATCH_SIZE}, "
             f"таймаут {timeout} с, max_tokens {cfg.openrouter_max_output_tokens}, "
             f"reasoning {cfg.openrouter_reasoning_effort}.")

    t0 = time.monotonic()
    with httpx.Client(timeout=timeout) as client:
        ctx = RunContext(client=client, url=url, headers=headers, model=model,
                         max_tokens=cfg.openrouter_max_output_tokens,
                         effort=cfg.openrouter_reasoning_effort, t0=t0, sink=sink, masker=masker)
        # окно: 3 рабочих потока берут задания из очереди по порядку, без повторов и без пула ключей
        with ThreadPoolExecutor(max_workers=WINDOW) as pool:
            futures = [pool.submit(send_batch, i, b, ctx) for i, b in enumerate(batches, start=1)]
            records = [f.result() for f in futures]
    total = time.monotonic() - t0

    records.sort(key=lambda r: r["batch"])
    summary = summarize(records, total)
    text = summary_text(model, summary, records)
    payload = {"model": model, "timeout_sec": timeout, "window": WINDOW,
               "summary": summary, "records": records}
    (out_dir / JSON_NAME).write_text(
        json.dumps(masker.obj(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / TXT_NAME).write_text(masker.text(text) + "\n", encoding="utf-8")
    sink.say("")
    sink.say(text)
    sink.say(f"Файлы в папке результатов: {JSONL_NAME}, {JSON_NAME}, {TXT_NAME}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
