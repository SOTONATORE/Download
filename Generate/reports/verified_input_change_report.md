# Отчёт: вход workflow `verified` и аргумент CLI `--verified`

Дата: 14.10.2026. Статус: изменения внесены в рабочую копию, не закоммичены. Проверку выполняет пользователь: `generate-tests` и workflow на GitHub. Результаты локальных прогонов в этом отчёте не приводятся. Живых вызовов Vast, GHCR, Gemini, Hugging Face и GitHub не было; workflow не запускался; commit и push не выполнялись.

Цель: ручной переключатель `verified` на форме Generate. Тестовый запуск может идти с `verified: false`, обычный запуск остаётся с `true`. Умолчание `true` не изменено, `config.default.yaml` не менялся.

## 1. Изменённые файлы

| Файл | Что изменено |
|---|---|
| `.github/workflows/generate.yml` | В `on.workflow_dispatch.inputs` добавлен вход `verified` (раздел 2). В шаге `Run generation`: переменная окружения `VERIFIED: ${{ inputs.verified }}` после `MAX_INET_PRICE`; строка `--verified "$VERIFIED"` в массиве `args` после `--max-inet-price-per-tb`. Остальные входы и шаги не менялись: `Preflight check`, `Kill rented cards`, `Finalize release on completion or cancel`, `Upload run log`. |
| `Generate/cli.py` | Новая функция `_parse_verified(value)` рядом с `_balance_preflight`: пустая строка даёт `None`; `true` и `false` без учёта регистра дают `bool`; любое иное значение вызывает `argparse.ArgumentTypeError` с русским текстом (раздел 3). В `build_parser()`, подкоманда `run`: аргумент `--verified` с `type=_parse_verified` и `default=None`. В `cmd_run()`: в словарь `overrides` добавлена пара `"verified": args.verified`. Подкоманды `check`, `kill-cards`, `finalize-release` не менялись. |
| `Generate/tests/test_cli.py` | Новый блок тестов после `test_run_verified_false_from_config_reaches_offer_filter`: 7 тестовых функций, 14 тестовых случаев с учётом параметров. Разбор значений `true`, `false`, `TRUE`, `False`, пустой строки; отсутствие флага даёт `None`; `maybe`, `1`, `yes`, `tru` дают код 2 и русское сообщение, `run_generation` не вызывается; `--verified false` доводит `verified is False` до загруженного конфига; без флага конфиг даёт `True`; пустой вход не переопределяет `verified: false` из файла; `--verified TRUE` переопределяет `false` из файла. Для проверки загруженного конфига используется шпион на `cli._load_cfg`. |
| `Generate/SPEC.md` | Раздел 3, список входов `generate.yml` (раздел 4 этого отчёта). Строка журнала от 2026-10-14 первой в списке «v0.3» (раздел 4). |
| `Generate/reports/verified_input_change_report.md` | Этот отчёт (новый файл). |

Не менялись: `Generate/config.default.yaml`, `Generate/core/config_loader.py`, `Generate/core/orchestrator.py`, `Generate/core/vast_client.py`, другие workflow, другие тесты.

Как работает переопределение. `load_config` накладывает на файл только значения, которые не равны `None` (`core/config_loader.py`, строка с `if v is not None`). Поэтому отсутствие флага и пустой вход не трогают конфиг. Значение `verified` проверяется как логическое (`_check_value`): строка `"true"` отвергается, CLI передаёт именно `bool`.

## 2. Новый вход в форме workflow

- Имя: `verified`
- Тип: `string`. Не `boolean`: так пустое значение означает «взять из конфига».
- Умолчание: `"true"`
- Описание: «Только проверенные хосты Vast (true/false). По умолчанию true»

## 3. Сообщение об ошибке для недопустимого значения

Входные данные для примера: `run --verified maybe`. Вывод argparse, код завершения 2, трейсбека нет:

```
python -m Generate.cli run: error: argument --verified: допустимы только значения true или false (без учёта регистра); пустое значение означает «взять из конфига»
```

Строки `usage:` и префикс `error:` выводит стандартный argparse на английском, как у всех ошибок аргументов этого CLI. Текст проекта, то есть сообщение о допустимых значениях, на русском. Введённое значение в сообщение не выводится (как в `_read_log_level`), поэтому секреты и произвольный текст в лог не попадают.

## 4. Изменения SPEC.md

- Раздел 3, абзац «Входы `generate.yml`»: добавлен `verified`. Описание: `true` по умолчанию, только проверенные хосты Vast; `false` — любые хосты; пустое значение = взять из конфига; действует на один запуск и переопределяет `verified` из `config.default.yaml`. Там же отмечено, что при входе по умолчанию правка файла не действует на запуски из workflow.
- Журнал изменений, `v0.3`, первая строка списка: «2026-10-14: вход workflow `verified` (`.github/workflows/generate.yml`, тип `string`, по умолчанию `"true"`) и аргумент CLI `--verified` команды `run` переопределяют ключ `verified` из `config.default.yaml` на один запуск. Допустимы `true` и `false` без учёта регистра; пустое значение конфиг не переопределяет; любое иное значение — ошибка аргументов (код 2). Умолчание `true` не изменено и `config.default.yaml` не менялся. Статус: офлайн-тесты в `Generate/tests/test_cli.py`; живой прогон не выполнялся (правило 0.3).»

Раздел 8.2 (пункт «Выбор хоста») не менялся.

## 5. Изменённые файлы (пути в репозитории)

- `.github/workflows/generate.yml`
- `Generate/cli.py`
- `Generate/tests/test_cli.py`
- `Generate/SPEC.md`
- `Generate/reports/verified_input_change_report.md` (новый)

## 6. Риски

1. **Пустое значение означает конфиг, а не «без фильтра».** Тот, кто ждёт «любые хосты» при пустом поле, получит `verified: true` из `config.default.yaml`. Для любых хостов нужно указать `false`. Подсказку в описании входа стоит сохранить или уточнить при следующей правке формы.
2. **Умолчание не меняет производственный запуск, но workflow теперь всегда передаёт флаг.** Вход по умолчанию равен `true`, как и конфиг, поэтому текущие запуски не меняются. Однако значение формы перекрывает `config.default.yaml`. Если позже поменять `verified` в файле на `false`, запуски из формы всё равно пойдут с `true`, пока вход не изменён. Если нужно, чтобы главным был конфиг, умолчание входа должно быть пустым. Это отдельное решение, в этой правке оно не принималось: умолчание задано по требованию.
3. **Пробелы не обрезаются.** Значение `" true"` даёт ошибку аргументов (код 2). Это строже, чем могло бы быть, и соответствует правилу «допустимы только true и false».
4. **Шаги, которые не ищут предложения, флаг не получают.** Шаг `Preflight check` (`check`) и `Finalize release` (`finalize-release`) не используют фильтр поиска карт, поэтому `--verified` им не передаётся.
