# Generate/ — вторая часть: генерация клипов по SRT

Проект создаёт видеоклипы по файлу `final.srt`: Gemini пишет промпт для каждого сегмента, GitHub Actions арендует одиночные GPU-карты на Vast.ai, на каждой карте работает воркер с headless ComfyUI, готовый клип сразу уходит в GitHub Release и удаляется с раннера. Пользователь получает ссылки на клипы в итоговом релизе. Запуск один раз вручную через workflow `generate.yml`.

Источник истины по договорённостям, архитектуре и статусам этапов: [`SPEC.md`](SPEC.md). Если README и SPEC расходятся, верен SPEC.

## Структура папки

- `core/` — ядро: разбор SRT, расчёт длины клипа, промпты Gemini, оркестратор, клиент Vast.ai, бюджет, выходной адаптер GitHub Release. Ядро не знает про GitHub Actions.
- `worker/` — код, который едет на карту: HTTP-сервис, клиент ComfyUI, Dockerfile и скрипты установки.
- `tests/` — тесты pytest, без сети и ключей.
- `tools/` — диагностические скрипты, которые только читают данные (например, `vast_probe.py`).
- `model_profiles/` — профили видеомодели: `models/<имя>/` с `profile.yaml`, графом `workflow.json`, привязками `bindings.json` и инструкцией для Gemini `prompts/style.md`; `presets/` — пресеты (фаза E, пока пусто).
- `cli.py` — точка входа: `python -m Generate.cli check | run | kill-cards`.
- `config.default.yaml` — значения по умолчанию (раздел 3 SPEC).

## Тесты

Тесты запускает владелец вручную: в GitHub Actions открывается workflow `generate-tests.yml` (кнопка Run workflow). Автоматического запуска тестов при push нет.

## Секреты

Используются только три секрета GitHub (Settings → Secrets and variables → Actions): `GEN_VAST_API_KEY`, `GEN_GEMINI_API_KEY`, `HF_TOKEN`. Они передаются в шаги workflow через переменные окружения и нигде не печатаются. Токен воркера создаётся заново при каждом запуске и нигде не хранится.

## Где лежат результаты

- Артефакт `generate-<run_id>` (каталог запуска: run/ с журналом debug.log и сводкой, cards/, service/, FILES.txt и generate-run.log — журнал окон шагов), срок хранения 30 дней. Консоль Actions показывает только ключевые события, ошибки и предупреждения; подробности — в run/debug.log.
- GitHub Releases: клип-релизы `clips-<хэш>` (клипы), служебный `run-<run_id>-service` (промпты, список пропусков, отчёт, логи) и итоговый `run-<run_id>-final` (файл `generated_links.txt` со ссылками на клипы).
