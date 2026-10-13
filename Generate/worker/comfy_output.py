"""Вывод ComfyUI в буфер логов воркера (SPEC 8.4, раздел 10).

setup.sh пишет stdout и stderr ComfyUI через tee: в контейнерный лог (как раньше) и в файл.
Воркер читает этот файл с начала, построчно, и отдаёт строки в буфер логов.
Строки проходят маскирование, из них убираются ANSI-коды цвета и они усекаются до 300 символов.
Текст строк ComfyUI не переводится (SPEC 0.5): к нему добавляется только русская пометка.
Только стандартная библиотека (SPEC 0.2).
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("Generate.worker.comfy_output")

COMFY_LINE_MAX = 300             # символов в одной строке буфера (вместе с «…»)
COMFY_LOG_LOGGER = "Generate.worker.comfy"  # логгер строк ComfyUI: в буфер, в контейнерный лог не дублируется
COMFY_MARK = "Вывод ComfyUI: "   # русская пометка перед строкой, текст строки не меняется
READ_CHUNK = 1024 * 1024         # за один проход читаем не больше 1 МиБ
MAX_PENDING = 8192               # длиннее без перевода строки — отдаём кусками
POLL_SEC = 0.5
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def clean_comfy_line(raw: str, mask: Callable[[str], str]) -> str:
    """Строка для буфера: последняя версия строки после «\\r» (прогресс), без ANSI-кодов,
    с маской секретов, усечённая до COMFY_LINE_MAX символов.

    Порядок важен: маска применяется до усечения, иначе секрет, разрезанный границей,
    не совпадёт с шаблоном и попадёт в лог частично.
    """
    text = _ANSI_RE.sub("", raw.rsplit("\r", 1)[-1]).rstrip()
    text = mask(text)
    if len(text) > COMFY_LINE_MAX:
        text = text[:COMFY_LINE_MAX - 1] + "…"
    return text


class ComfyOutputReader:
    """Читает файл вывода ComfyUI и передаёт готовые строки в sink.

    Строки поступают с начала файла (в том числе вывод запуска ComfyUI). Если файл усечён,
    чтение начинается заново. Частично записанная строка ждёт продолжения (до MAX_PENDING байт).
    Ошибка чтения не останавливает поток: одно предупреждение, затем повтор по расписанию.
    """

    def __init__(self, path, sink: Callable[[str], None], mask: Callable[[str], str], *,
                 poll_interval: float = POLL_SEC) -> None:
        self.path = Path(path)
        self._sink = sink
        self._mask = mask
        self._poll = poll_interval
        self._offset = 0
        self._pending = b""
        self._warned = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.lines_emitted = 0

    def poll_once(self) -> int:
        """Одна порция чтения. Возвращает число переданных строк."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return 0
        if size < self._offset:  # файл усечён или пересоздан
            self._offset = 0
            self._pending = b""
        if size == self._offset:
            return 0
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            data = fh.read(min(size - self._offset, READ_CHUNK))
        self._offset += len(data)
        parts = (self._pending + data).split(b"\n")
        self._pending = parts.pop()
        if len(self._pending) > MAX_PENDING:
            parts.append(self._pending)
            self._pending = b""
        emitted = 0
        for raw in parts:
            cleaned = clean_comfy_line(raw.decode("utf-8", errors="replace"), self._mask)
            if cleaned:
                self._sink(cleaned)
                emitted += 1
        self.lines_emitted += emitted
        return emitted

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # чтение не должно останавливаться
                if not self._warned:
                    self._warned = True
                    log.warning("Чтение вывода ComfyUI прервано ошибкой: %s. Повтор по расписанию.",
                                type(exc).__name__)
            self._stop.wait(self._poll)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="comfy-output", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
