"""Вывод ComfyUI в буфер логов воркера (SPEC 8.4, раздел 10).

setup.sh пишет stdout и stderr ComfyUI через tee: в контейнерный лог (как раньше) и в файл.
Воркер читает этот файл с начала и передаёт строки приёмнику (sink) по частям (этап 9в2):
части сразу уходят на диск, в памяти остаётся только хвост строки, ещё не записанный.
Строка без перевода строки длиннее 1 МиБ делится на части, на границе ставится пометка
«[строка продолжается]». Строки проходят маскирование, из них убираются ANSI-коды цвета.
Текст строк ComfyUI не переводится (SPEC 0.5): к нему добавляется только русская пометка.
Только стандартная библиотека (SPEC 0.2).
"""
from __future__ import annotations

import codecs
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("Generate.worker.comfy_output")

COMFY_LOG_LOGGER = "Generate.worker.comfy"  # логгер строк ComfyUI: в буфер, в контейнерный лог не дублируется
COMFY_MARK = "Вывод ComfyUI: "   # русская пометка перед строкой, текст строки не меняется
READ_CHUNK = 1024 * 1024         # за один проход читаем не больше 1 МиБ
POLL_SEC = 0.5
PIECE_LIMIT = 1024 * 1024        # символов строки без перевода строки; дальше пометка границы частей
PIECE_MARK = "[строка продолжается]"
HOLD_CHARS = 256                 # хвост строки, который не пишется, пока строка не продолжится
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Маски с пробелами внутри (Authorization, Bearer, X-Worker-Token) захватывают текст после ключа:
# часть строки не режется начиная с первого такого ключа, иначе значение попадёт в запись без маски.
_HOLD_RE = re.compile(r"authorization|bearer|x-worker-token", re.IGNORECASE)
_CUT_RE = re.compile(r"\s|\x1b")  # допустимые места разреза: после пробела или перед ESC-кодом


def clean_comfy_line(raw: str, mask: Callable[[str], str]) -> str:
    """Строка для буфера: последняя версия строки после «\\r» (прогресс), без ANSI-кодов,
    с маской секретов. Строка целиком, без усечения длины.
    """
    text = _ANSI_RE.sub("", raw.rsplit("\r", 1)[-1]).rstrip()
    return mask(text)


def safe_cut(text: str, final: bool) -> int:
    """Сколько символов начала text можно записать сейчас (0 = ничего).

    Разрез ставится только после пробельного символа или перед ESC-кодом: ни один маскируемый
    шаблон без пробелов (sk-, hf_, AIza, ghp_, значение токена) не переходит через разрез.
    Хвост из HOLD_CHARS символов не пишется, пока строка не продолжится. Шаблоны с пробелами
    (см. _HOLD_RE) держатся целиком до конца строки. Для финального разреза (конец строки) эти ограничения не нужны.
    """
    limit = len(text) if final else len(text) - HOLD_CHARS
    hold = _HOLD_RE.search(text)
    if hold is not None and not final:
        limit = min(limit, hold.start())
    if limit <= 0:
        return 0
    cut = 0
    for m in _CUT_RE.finditer(text, 0, min(len(text), limit + 1)):
        pos = m.start() if m.group() == "\x1b" else m.end()
        if 0 < pos <= limit:
            cut = pos
    return cut


class CallbackSink:
    """Приёмник для обычной функции: собирает части и отдаёт функции готовую строку целиком."""

    def __init__(self, fn: Callable[[str], None]) -> None:
        self._fn = fn
        self._parts: list = []

    def write_piece(self, text: str) -> None:
        self._parts.append(text)

    def cancel_line(self) -> None:
        self._parts = []

    def end_line(self) -> None:
        line = "".join(self._parts)
        self._parts = []
        self._fn(line)


class ComfyOutputReader:
    """Читает файл вывода ComfyUI и передаёт строки приёмнику sink по частям.

    sink — объект с методами write_piece(text), cancel_line() и end_line() (см. server._LineSink)
    или обычная функция (тогда получает строку целиком). Строки поступают с начала файла
    (в том числе вывод запуска ComfyUI). Если файл усечён, чтение начинается заново.
    Частично записанная строка видна приёмнику, пока не закрыта; строка с «\\r» заменяется
    последним состоянием (как раньше): записанные части отменяются через cancel_line().
    Ошибка чтения не останавливает поток: одно предупреждение, затем повтор по расписанию.
    """

    def __init__(self, path, sink: Any, mask: Callable[[str], str], *,
                 poll_interval: float = POLL_SEC) -> None:
        self.path = Path(path)
        self._sink = sink if hasattr(sink, "write_piece") else CallbackSink(sink)
        self._mask = mask
        self._poll = poll_interval
        self._offset = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._cur = ""          # ещё не записанный хвост текущей строки (ограничен PIECE_LIMIT + HOLD_CHARS + порцией)
        self._wrote = False     # часть текущей строки уже передана приёмнику
        self._raw_written = 0   # сколько исходных символов текущей строки уже записано (для пометки границы)
        self._warned = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.lines_emitted = 0

    def poll_once(self) -> int:
        """Одна порция чтения. Возвращает число закрытых строк, переданных приёмнику."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return 0
        if size < self._offset:  # файл усечён или пересоздан
            self._reset_line()
            self._offset = 0
            self._decoder.reset()
        if size == self._offset:
            return 0
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            data = fh.read(min(size - self._offset, READ_CHUNK))
        self._offset += len(data)
        text = self._decoder.decode(data)
        emitted = 0
        pos = 0
        while True:
            idx = text.find("\n", pos)
            if idx < 0:
                break
            self._add_text(text[pos:idx])
            if self._end_line():
                emitted += 1
            pos = idx + 1
        self._add_text(text[pos:])
        self._flush(final=False)
        self.lines_emitted += emitted
        return emitted

    def _reset_line(self) -> None:
        if self._wrote:
            self._sink.cancel_line()
        self._cur = ""
        self._wrote = False
        self._raw_written = 0

    def _add_text(self, s: str) -> None:
        if "\r" in s:  # строка заменяется последним состоянием после «\r» (прогресс)
            self._reset_line()
            s = s.rsplit("\r", 1)[-1]
        self._cur += s

    def _end_line(self) -> bool:
        """Закрывает строку. True, если строка передана (пустые строки не передаются, как раньше)."""
        self._flush(final=True)
        emitted = self._wrote
        if emitted:
            self._sink.end_line()
        self._cur = ""
        self._wrote = False
        self._raw_written = 0
        return emitted

    def _flush(self, final: bool) -> None:
        """Передаёт приёмнику всё, что можно записать безопасно (см. safe_cut)."""
        while self._cur:
            cut = safe_cut(self._cur, final)
            if cut == 0:
                if final:
                    cut = len(self._cur)
                elif len(self._cur) > PIECE_LIMIT:
                    cut = len(self._cur) - HOLD_CHARS  # строка без безопасного разреза дольше 1 МиБ: жёсткий разрез
                else:
                    return
            part, self._cur = self._cur[:cut], self._cur[cut:]
            self._write_part(part, last=final and not self._cur)

    def _write_part(self, part: str, *, last: bool) -> None:
        text = _ANSI_RE.sub("", part)
        if last:
            text = text.rstrip()
        cleaned = self._mask(text)
        if cleaned:
            self._sink.write_piece(cleaned)
            self._wrote = True
        before = self._raw_written // PIECE_LIMIT
        self._raw_written += len(part)
        if self._wrote and self._raw_written // PIECE_LIMIT > before:
            self._sink.write_piece(PIECE_MARK)  # граница части: строка продолжается

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
