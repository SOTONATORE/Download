# core/gemini_prompts.py (этап 4а)

Субтитры (SRT) -> промпты для видеомодели через Gemini. Здесь: данные, запрос, ответ, проверка.
Не здесь: лимитер RPM/TPM, счётчик RPD, цепочка моделей, `prompts.json` (см. раздел «Этап 4б», модуль `core/gemini_orchestrator.py`).

## Входы

| Что | Описание |
|---|---|
| `segments` | `list[Segment]` из `srt_parser` по ПОЛНОМУ SRT |
| `wanted_nums` | номера из `missing.txt` (дубли убираются, порядок по возрастанию) |
| `cfg: PromptConfig` | `style_file` (путь к `prompt_styles/<модель>.md` из профиля), `style_brief`, `max_prompt_chars` (1500), `batch_size` (20), `response_format` (по умолчанию `[{"segment_index": int, "prompt": str}]`; значения: типы Python или строки `"int"`/`"str"`) |
| `transport` | объект с методом `generate_json(model, system, user, schema) -> str` (`GeminiTransport`) |
| `model` | API-ID модели, параметром. В коде не зашит |

## Публичные сущности

- `build_requests(segments, wanted_nums, cfg) -> list[BatchRequest]`. Предложения и Before/After считаются по всему SRT, затем берутся только нужные номера. Номер идёт в запрос только строкой `segment_index: N`; модели велено не писать его в текст промпта.
- `generate_prompts(segments, wanted_nums, cfg, transport, model, on_batch=None) -> list[PromptResult]`. Один батч = один запрос; при нарушениях один retry только по плохим номерам; остаток получает `needs_review=True` (не статус). Номер без пригодного текста: `prompt=""`, `needs_review=True`.
- `parse_and_validate(raw, expected_nums, cfg) -> (results, problems)`. `results`: `dict[num, PromptResult]`; `problems`: `list[Problem(num, reason)]`.
- `PromptResult(num, prompt, needs_review, review_reason, extra)`; `extra` хранит дополнительные поля из `response_format` профиля.
- `strip_user_only(text)`: вырезает блоки `<!-- USER_ONLY_START -->...<!-- USER_ONLY_END -->`; непарный маркер вызывает `GeminiInputError`.
- `merge_segments_into_sentences(segments)`: склейка в предложения.
- `RealTransport(api_key_env=..., timeout=120, client=None)`: httpx. `close()` и `with` закрывают только внутренний клиент (повтор безопасен); переданный снаружи `client` не закрывается. Ключ читается из окружения при создании (имя: `gp.ENV_KEY_NAME`), уходит только в заголовок `x-goog-api-key`. Запрос: `POST {base}/{model}:generateContent`, `generationConfig.responseMimeType="application/json"`, `generationConfig.responseJsonSchema`.

## Что проверяет валидатор

Ровно нужные номера (нет пропусков, дублей; лишние игнорируются с предупреждением), целый `segment_index`, непустой `prompt`, длина <= `max_prompt_chars`, нет мусора: markdown-ограждения и заголовки; служебные формы `segment_index`, «Segment 4», «segment #4», «Segment:» в начале (обычное слово «segment» допустимо); номер-метка в начале (`[3]`, `12: `, `3. `, `4) `, `12 - `), но не «35-year-old», «2.5 m», «12:30»; обломки JSON. Переводы строк в промпте заменяются пробелами (один абзац).

## Ошибки

| Исключение | `exit_code` | Когда |
|---|---|---|
| `GeminiInputError` | 2 | номер вне SRT, пустой список, пустой SRT, плохой конфиг, непарные USER_ONLY, не задан ключ. Всегда до первого обращения к transport |
| `GeminiRuntimeError` | 3 | сеть, HTTP != 200. Поля: `status`, `code`, `daily_quota`, `retry_after` (для 4б) |

Текст `GeminiRuntimeError` строится только из безопасных полей (статус, код, имя класса сетевой ошибки). Тело ответа, заголовки и запрос в него не попадают; цепочка исключений обрывается (`from None`). Логгеры `httpx`/`httpcore` ставятся на WARNING. Логи на русском, тексты промптов в логи не пишутся.

## Колбэк `on_batch` (подключение в 4б)

Сигнатура: `on_batch(results: list[PromptResult]) -> None`, необязательный последний параметр `generate_prompts`.

- Момент вызова: после каждого готового батча (с учётом retry), до запроса следующего. Один батч = один вызов.
- Состав `results`: `PromptResult` этого батча по возрастанию номера, включая `needs_review=True` (в том числе с `prompt=""`). Номеров других батчей нет.
- При `GeminiRuntimeError` на батче N колбэк уже получил батчи 1..N-1; сам сбойный батч не отдаётся.
- Исключения колбэка не глотаются: они выходят из `generate_prompts`, следующие батчи не запрашиваются.
- Без колбэка поведение прежнее.

Типичное подключение: в колбэке дописывать результаты в `prompts.json` (атомарно) и обновлять счётчики; цепочку моделей и лимитер вешать на `transport.generate_json`. После сбоя пересчитать `wanted_nums` по `prompts.json` и повторить запуск.

Поведение транспорта при 200 без `candidates` или без текста: `generate_json` возвращает пустую строку (не исключение); батч получает retry, остаток помечается `needs_review`. Не-200 с телом не в JSON даёт `GeminiRuntimeError` с `status` и пустым `code`.

## Для 4б

Оборачивайте `transport.generate_json` лимитером и цепочкой моделей; различайте суточную квоту (`daily_quota=True`) и RPM (429 без этого признака, ждать `retry_after`). Прогон батчей лежит в `generate_prompts`; для идемпотентности передавайте в `wanted_nums` только номера без готового промпта.

## Сверка с документацией

- Подтверждено живым запросом 2026-10-05 (GitHub Actions, секрет `GEN_GEMINI_API_KEY`): `POST /v1beta/models/gemini-3.5-flash-lite:generateContent`, заголовок `x-goog-api-key`, `generationConfig.responseMimeType="application/json"` и `responseJsonSchema` дают HTTP 200 и JSON по схеме. Ответ: `candidates[0].content.parts[].text`. Модель с неверным именем даёт 404, `error.status="NOT_FOUND"`.
- Документация Google (ai.google.dev/gemini-api/docs/structured-output) описывает Interactions API; он тоже отвечает, но мы его не используем.
- Подтверждено страницей: ID `gemini-3.5-flash-lite` и `gemini-3.1-flash-lite` (ai.google.dev/gemini-api/docs/models/...), заголовок `x-goog-api-key`.
- НЕ подтверждено, проверить живым 429 в Actions: признак суточной квоты (`quotaId` с `PerDay` в `details[].violations[]`) и формат `retryDelay` (`"34s"`). Пока признак не найден, 429 считается минутным (RPM).

## Чек-лист живой проверки

- [ ] Получить живой 429 в Actions и сверить: `quotaId` с `PerDay` в `details[].violations[]` (суточная квота) и формат `retryDelay` (`"34s"`). Пока не подтверждено, 429 без признака суточной квоты считается минутным (RPM).


# Этап 4б: `core/gemini_orchestrator.py`

Обёртка над 4а: лимиты, цепочка моделей, `prompts.json`. Только стандартная библиотека и `gemini_prompts`. Ключ здесь не читается и не упоминается.

## Точка входа

`generate_and_store(segments, wanted_nums, cfg, inner_transport, prompts_path, *, profile, timing, models=DEFAULT_MODELS, rpm_limit=15, tpm_limit=250000, rpd_limit=500, rpd_used=None, output_reserve_tokens=2000, max_rate_retries=6, clock, sleep) -> dict`

- `timing: TimingParams(fps, frame_rule, clip_min_sec, clip_max_sec, gpu_max_sec, tail_pad_sec)`: параметры для `timing.compute_timings` (считаются по ПОЛНОМУ SRT).
- `inner_transport`: `RealTransport` или заглушка; оборачивается в `ManagedTransport`.
- `DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")`. `gemini_models` в `config.default.yaml` пока пуст, значения нужно передавать из конфига.
- Возвращает итоговое состояние (то, что лежит в `prompts.json`).

## Порядок работы

1. Проверка входа (пустой список, номера вне SRT): `GeminiInputError`, без обращения к Gemini.
2. Если `prompts.json` есть: `srt_hash` сверяется с текущим SRT. Несовпадение, битый JSON или неверная структура дают `GeminiInputError` (код 2), квота не тратится.
3. К генерации идут только номера из `wanted_nums` без готового промпта (`status="ready"` и непустой `prompt`). Если таких нет, Gemini не вызывается и файл не трогается.
4. После каждого батча (`on_batch`) запись обновляется и `prompts.json` пишется атомарно: временный файл в той же папке, `flush`, `fsync`, `os.replace`. При сбое старый файл цел, временный удаляется.

## Формат записи

`num, start_ms, end_ms, clip_ms, num_frames` (из `compute_timings`), `prompt`, `status`, `attempts`, `error`, `needs_review`, `extra`. Верхний уровень: `srt_hash`, `model`, `profile`, `prompts` (ключи по возрастанию номера).

- `status`: `"ready"` при непустом промпте, иначе `"failed"` (такой номер будет сгенерирован заново при следующем запуске).
- `attempts`: +1 за каждый запуск, где номер отправлялся в Gemini (внутренний retry 4а не считается).
- `error`: `review_reason` из 4а. `needs_review=True` с непустым промптом считается готовым и повторно не отправляется.
- `model`: модель, давшая последний сохранённый батч. Модель по каждой записи отдельно не хранится.

## Лимитер и цепочка моделей

- `RateLimiter`: скользящее окно 60 с, RPM и TPM. Токены оцениваются как `(len(system)+len(user))/3 + output_reserve_tokens`. Запрос крупнее TPM не блокируется навсегда (оценка усекается до TPM). У каждой модели свой лимитер.
- RPD: счётчик запросов на модель (считаются все отправленные, включая 429). Дошёл до `rpd_limit`: модель считается исчерпанной.
- `GeminiRuntimeError(daily_quota=True)`: модель исчерпана, переход к следующей в цепочке, запрос повторяется.
- 429 без `daily_quota`: пауза `retry_after` (иначе 5, 10, 20... с, потолок 60), затем повтор на той же модели. Пауза ограничена диапазоном 1..120 с. После `max_rate_retries` подряд ошибка пробрасывается.
- Остальные ошибки пробрасываются как есть.
- Исчерпаны все модели: `GeminiRuntimeError` (код 3, `status=429`, `daily_quota=True`). Готовые батчи уже сохранены, после сброса квоты запуск продолжится с места остановки.
- Параметр `model`, который `generate_prompts` передаёт в транспорт, игнорируется: модель выбирает цепочка.

## Ограничения

- Счётчик RPD живёт в памяти процесса. Между запусками он не сохраняется (в формате `prompts.json` для него нет поля); при необходимости передавайте `rpd_used={модель: число}`. Если суточная квота кончилась, это обнаружится по ответу 429.
- Признак суточной квоты в `RealTransport` всё ещё не подтверждён живым 429 (см. чек-лист выше). Пока он не найден, исчерпание квоты выглядит как RPM-429 и после `max_rate_retries` завершится ошибкой, а не сменой модели.
