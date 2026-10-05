# Generate/core: srt_parser.py и timing.py

Чистые модули (только stdlib, без сети/БД/настроек). Время везде в целых мс.
Импорт: `from Generate.core.srt_parser import ...`, `from Generate.core.timing import ...`.
Тесты лежат в `Generate/tests/`; запуск из корня репозитория: `python -m pytest Generate/tests` (см. `Generate/README.md`).

## srt_parser.py
- `Segment(num, start_ms, end_ms, text)`; `SrtError(ValueError)`.
- `parse_srt(path)` / `parse_srt_text(raw)` -> `list[Segment]`, отсортирован по `num`.
  Поддержаны BOM, `\r\n`/`\n`, разделители `,` и `.`. Теги `<..>` и `{..}` убираются, пробелы схлопываются.
- Ошибки (`SrtError`, с номерами): пустой файл, нераспознанный блок, `end < start`, дубли номеров, пропуски номеров.
- `check_srt(segments)` -> `list[str]` предупреждений: перекрытие, пустой текст, нулевая длина.
- `srt_hash(segments)` -> sha256 по `num|start|end|text` (каждый сегмент с `\n` в конце).

## timing.py
- `compute_timings(segments, fps, frame_rule, clip_min_sec, clip_max_sec, gpu_max_sec, tail_pad_sec) -> list[ClipTiming]`
- `frames_for_rule(raw_ms, fps, frame_rule) -> int`

Формула:
1. `raw_i = start_{i+1} - start_i`; для последнего `raw = (end - start) + tail_pad`. Паузу отдельно не добавлять.
2. `need = ceil(raw_ms * fps / 1000)` в целых `(raw*fps + 999)//1000`, минимум `min_frames`.
3. `num_frames` = наименьшее допустимое по правилу значение `>= need`.
4. `clip_ms = round(num_frames * 1000 / fps)` (целочисленно). Всегда `clip_ms >= raw_ms`.

Если `raw <= 0`: берётся `end - start` сегмента, если и это `<= 0` — `clip_min_sec`; плюс предупреждение.
`ClipTiming.raw_ms` содержит фактически использованную длину (после такой подмены).
Предупреждения: «короче минимума», «длиннее рекомендуемого максимума», «выше максимума, который тянет карта»
(при превышении `gpu_max_sec` выдаются оба сообщения: о максимуме и о карте). Клип остаётся в результате.

## Как добавить правило кадров
В `Generate/core/timing.py` добавить одну функцию:

```python
@frame_rule("my_rule")
def _rule_my(need: int, rule: dict) -> int:
    # вернуть наименьшее допустимое число кадров >= need (need уже >= min_frames)
    ...
```
Правила: `"8k+1"` (шаг 8 по умолчанию, можно `"n"`), `"nk+1"` (обязателен `"n"`), `"any"`.
Параметры берутся из словаря профиля: `{"kind": "nk+1", "n": 4, "min_frames": 5}`.
Неизвестный `kind` -> `ValueError`.
