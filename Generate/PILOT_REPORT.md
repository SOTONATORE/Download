# Пилот ComfyUI на Vast.ai: отчёт об изменениях

Цель: проверить инфраструктуру Vast (поиск, аренда, ожидание `running`, ответ порта 8188, уборка, бюджет)
без нашего воркера. Боевой путь `generate.yml` и `Generate/worker/` не меняются.

Проверка в этой сессии: только чтение кода, `py_compile` / `ast.parse` для изменённых модулей и разбор YAML.
Тесты, workflow, живые вызовы Vast / GHCR / Hugging Face / Gemini / GitHub API не запускались.

## 1. Изменённые и новые файлы

| Файл | Тип | Строки (новый файл) | Что изменено |
|---|---|---|---|
| `.github/workflows/pilot-comfyui.yml` | новый | 1–217 | копия `generate.yml` с отличиями (см. раздел 2) |
| `Generate/core/orchestrator.py` | изменён | 64–72 | константы пилота: порт 8188, команда старта (UNVERIFIED), путь и таймаут пробы |
| `Generate/core/orchestrator.py` | изменён | 205–229 | функция `_mapped_port` (внешний порт из поля `ports` экземпляра) |
| `Generate/core/orchestrator.py` | изменён | 272–279 | параметры конструктора `pilot`, `pilot_start_cmd` (по умолчанию выключено) |
| `Generate/core/orchestrator.py` | изменён | 303–311 | метка `pilot-<run_id>` при `pilot=True`, иначе `gen-<run_id>` как раньше |
| `Generate/core/orchestrator.py` | изменён | 438–445 | `run()`: ветка пилота до `parse_missing`, Release и Gemini |
| `Generate/core/orchestrator.py` | изменён | 499–611 | `_run_pilot()` и `_probe_pilot()`: цикл пилота, замер времени, уборка |
| `Generate/core/orchestrator.py` | изменён | 756–787 | `_rent_one`: env только с портом 8188 в пилоте; `onstart` в пилоте; основной env без изменений по ключам |
| `Generate/cli.py` | изменён | 364–382 | `_pilot_mode()` и проверка: пилот требует явный образ |
| `Generate/cli.py` | изменён | 424–432 | в пилоте не создаются Gemini-транспорт и Release-клиент |
| `Generate/cli.py` | изменён | 488–495 | в `run_generation` передаются `pilot`, `pilot_start_cmd` |
| `Generate/cli.py` | изменён | 661–670 | флаги `--pilot` и `--pilot-start-cmd` |
| `Generate/PILOT_REPORT.md` | новый | — | этот отчёт |

Не менялись: `.github/workflows/generate.yml` (побайтно идентичен архиву), `docker_build.yml`,
`Generate/worker/*`, `Generate/core/vast_client.py`, `budget.py`, `bad_hosts.py`, тесты.
Папки `Download/` в архиве нет (есть `Download-main/`); её не трогали.

## 2. Отличия pilot-comfyui.yml от generate.yml

- Вход `docker_image`: по умолчанию пусто; при пустом значении берётся переменная репозитория `PILOT_IMAGE`
  (`vars.PILOT_IMAGE`). Если и её нет, шаг завершается кодом 2 до любого вызова Vast.
- Новый вход `pilot_start_cmd` (пусто = `PILOT_START_CMD` из кода).
- Шаг «Preflight check» (docker login в GHCR и `check --docker-image --worker-build`) удалён. Образ воркера и тег = SHA не используются.
- `WORKER_BUILD` и `--worker-build` удалены из шага запуска.
- Метка карт `pilot-<run_id>`; шаг уборки: `kill-cards --label "pilot-$RUN_ID"`. Без этого уборка
  по умолчанию искала бы `gen-<run_id>` и не нашла бы карты пилота.
- Не передаются `HF_TOKEN`, `GEN_GEMINI_API_KEY`, `GH_TOKEN`: веса не грузятся, Gemini и Release не используются.
- Шаг «Finalize release» не включён: пилот не пишет в Release.
- `permissions`: `packages: read` убран, `contents: write` заменён на `read`.
- Отдельная группа `concurrency` (`pilot-comfyui`). Пилот не блокирует и не отменяет `generate`.
- Таймаут задания 360 мин, как в `generate.yml`. Внутренние сроки пилота (молчание 20 мин, жизнь карты 90 мин) берутся из конфига.
- Входы, которые пилот не использует (`srt_path`, `missing_path`, `limit_clips`, `release_tag`, `max_cards`,
  `model_profile`, `style_brief`), в pilot-файл не включены. CLI получает для них значения по умолчанию из конфига.

## 3. Что не удалось проверить без живых вызовов (UNVERIFIED)

1. Образ ComfyUI: какой образ использовать и есть ли в нём ENTRYPOINT/CMD. В репозитории нет ни одного
   образа ComfyUI, кроме `Generate/worker/Dockerfile`, который использует `ENTRYPOINT setup.sh`. Этот
   setup.sh требует `HF_TOKEN`, качает веса и запускает `server.py`. Для пилота этот образ не подходит.
2. Команда старта: `PILOT_START_CMD = "python /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 8188"`.
   Путь `/workspace/ComfyUI` взят из нашего Dockerfile и для внешнего образа не подтверждён.
3. Механизм запуска команды: передаётся через параметр `onstart` `create_instance`. В репозитории нет ни одного
   места, где этот параметр используется (только сигнатура). Неизвестно, заменяет ли `onstart` команду образа
   или выполняется после неё, и выполняется ли он до старта процесса.
4. Ключ проброса порта `"8188/tcp"` в поле `ports` ответа Vast: предполагается тот же формат, что у `"8000/tcp"`.
   В репозитории подтверждён только формат для 8000.
5. Доступность `public_ipaddr` + проброшенного порта 8188 с нашего раннера. Не проверялось.
6. Поведение `GET /system_stats` у выбранного образа ComfyUI (в нашем worker setup.sh этот эндпоинт используется для проверки готовности).
7. Существует ли в настройках репозитория переменная `PILOT_IMAGE`.
8. Поведение `RealTransport` и `GhReleaseStore` без ключей: в пилоте они не создаются, поэтому это не влияет на пилот.
   Но `tests` могут проверять их создание; не проверялось.

## 4. Риски для боевого пути (`generate.yml`) и как проверить без запуска

Общий принцип: все изменения в общих функциях защищены условиями `if self.pilot` / `_pilot_mode(args)`,
а в основной путь добавляется только `onstart=None`, который в `create_instance` не попадает в payload (`if onstart:`).

| Место | Риск | Как проверить без запуска |
|---|---|---|
| `_rent_one`, блок env (756–770) | Переструктурирован блок с `WORKER_TOKEN`, `-p 8000:8000` и `HF_TOKEN`. Ключи и порядок сохранены, но блок теперь в `else`. | `diff -u` с оригиналом: в `else` ровно прежние строки с отступом +4. Тесты env в `test_orchestrator.py` и `test_vast_client.py` (проверяют наличие `WORKER_TOKEN`, `HF_TOKEN`, `-p 8000`). |
| `_rent_one`, `create_instance` (783) | Добавлен kwarg `onstart`. Фейки в тестах уже принимают `onstart=None`. Реальный `VastClient.create_instance` принимает его по сигнатуре. | Тест payload в `test_vast_client.py` (строки 264–290): поле `onstart` не должно появляться при `None`. |
| Метка (308) | При `pilot=False` метка `gen-…`, как раньше. | `grep -n "gen-" Generate/tests` и сравнение с выводом `Orchestrator.label` для обычного запуска. |
| `run()` (441–442) | Ветка стоит до `parse_missing`, но только при `self.pilot`. | Чтение кода: при `pilot=False` первая строка `try` не меняется по смыслу. |
| `cli.cmd_run` (379, 428–429) | `_pilot_mode(args)` возвращает False, если атрибута нет. Транспорт и Release создаются как раньше. | `test_cli.py`: вызовы `cmd_run` с обычными `Namespace`. |
| `run_generation(...)` (491–492) | Новые kwargs. Фейк `fake_run` в тестах принимает `*args, **kwargs` (строки 181, 733). Строгая сигнатура сломала бы тест. | Проверить `test_cli.py` (строки 181, 389) и `test_vast_client.py` (733). |
| `cli.kill-cards` | Не менялся. Для боевого запуска `--run-id` по-прежнему строит `gen-<run_id>`. | Нет изменений в коде. |

Команда для проверки на машине ревьюера (НЕ запускалась здесь): `pytest Generate/tests -q`. Особое внимание к `test_orchestrator.py`, `test_vast_client.py`, `test_cli.py`.

## 5. Лог-маркеры по шагам

| Шаг | Что ожидать в логе | Откуда |
|---|---|---|
| Install dependencies | `pip install` без ошибок | workflow |
| Проверка Vast (до аренды) | `Эндпоинт списка экземпляров Vast отвечает.` и `Кредит Vast.ai: … USD, минимум … USD.` | `cli._vast_preflight` (без изменений) |
| Аренда | `Арендована карта <id> по предложению <offer> (слот 0, поколение 1, …): … USD/ч.` | `_rent_one` (без изменений) |
| Состояние Vast | `Карта <id>: состояние Vast «loading»`, затем «created», затем «running» | `_run_pilot`, строка при смене статуса |
| Статус `running` | `PILOT-МАРКЕР running: карта <id> через N с после аренды.` | `_run_pilot` |
| Ответ порта 8188 | `PILOT-МАРКЕР порт 8188: HTTP 200 через N с после аренды.` | `_run_pilot` |
| Ожидание порта | в режиме DEBUG: `PILOT: порт 8188 пока не отвечает (код …).` | `_run_pilot` |
| Таймаут порта | `PILOT: порт 8188 не ответил за 20 мин после аренды.` (ERROR) | `_run_pilot` |
| Таймаут аренды | `PILOT: карту не удалось арендовать за 20 мин.` (ERROR) | `_run_pilot` |
| Уборка пилота | `Карта <id>: адрес воркера неизвестен, логи не запрошены.` (ожидаемо: у пилота нет воркера) и `Уничтожено карт по метке pilot-<run_id>: 1.` | `_cleanup` |
| Итог | `PILOT-ИТОГ: результат=ok; аренда→running=N с; аренда→ответ 8188=N с; карт арендовано 1, уничтожено 1; уборка выполнена.` | `_run_pilot` |
| Бюджет | `Бюджет запуска: режим аренды …; по картам: карта <id>: X USD; всего X USD из лимита … USD.` | `_log_budget` |
| Kill rented cards | `Уничтожено карт: N (метка pilot-<run_id>).` | `cli.cmd_kill_cards` |

Время от аренды до `running` и до ответа 8188 записывается в строку `PILOT-ИТОГ` (целые секунды). Строки с префиксом `PILOT` попадают в «Итоги пилота» (шаг summary) и в артефакт `pilot-run-log-<run_id>`.

Критерий успеха: `result=ok` означает, что статус `running` и `HTTP 200` на `GET /system_stats` порта 8188 получены, после чего карта уничтожена. Ответ с другим кодом или сетевая ошибка не считаются успехом (код ответа виден только в DEBUG).

## 6. Прочие замечания

- В пилоте карта уничтожается сразу после успеха. Это сделано, чтобы не платить за простой.
- Пилот использует `rental_type` и цены по умолчанию (прерываемая, bid). Это влияет на стоимость и на шанс, что карту заберёт хост.
- Пробный запуск лучше делать с `budget_limit_usd` около 0.5–1.0 USD, чтобы ограничить расход при сбое.
- Метки разделяют карты пилота и боевого запуска. Очистка боевого `generate` не затрагивает карты пилота, и наоборот.

## 7. Изменения для vastai/comfy

Разделы 1–6 описывают прежний путь с портом 8188. Для образа `vastai/comfy:v0.37.0-cuda-13.2-py312` действует этот раздел: образ сам запускает ComfyUI через supervisor, внутри контейнера ComfyUI слушает порт **18188**, внешний порт выдаёт Vast. Пилот не переопределяет команду старта и не передаёт токены. Боевой путь `generate` не меняется.

### 7.1. Изменённые строки (номера в текущих файлах)

| Файл | Строки | Что изменено |
|---|---|---|
| `Generate/core/orchestrator.py` | 68 | `PILOT_HTTP_PORT = 18188` (было 8188) |
| `Generate/core/orchestrator.py` | 69–71 | константа команды старта с путём `main.py` удалена; `PILOT_PROBE_PATH` сохранён; добавлен `PILOT_AUTH_CODES = {401, 403}` |
| `Generate/core/orchestrator.py` | 227–249 | новые функции `_pilot_verdict` (код пробы → ok / auth_required / ждать) и `_port_keys_text` (только имена ключей `ports`) |
| `Generate/core/orchestrator.py` | 330–331 | пустая `pilot_start_cmd` становится `None`; значения по умолчанию из кода больше нет |
| `Generate/core/orchestrator.py` | 527–641 | `_run_pilot`: проба на 18188 по внешнему порту из `ports`, вердикты, маркеры, итог `PILOT-ИТОГ` |
| `Generate/core/orchestrator.py` | 805 | строка `-p …` в пилотном env теперь `-p 18188:18188` (через константу, сам текст строки не менялся) |
| `Generate/cli.py` | 492–493 | `--pilot-start-cmd`: пустое значение не передаётся дальше (`None`) |
| `Generate/cli.py` | 668 | текст подсказки для `--pilot-start-cmd` |
| `.github/workflows/pilot-comfyui.yml` | 15 | описание входа `pilot_start_cmd`: «пусто = не передавать, образ стартует сам» |
| `.github/workflows/pilot-comfyui.yml` | 69 | описание `silent_host_timeout_min`: порт 18188 |
| `Generate/tests/test_vast_client.py` | 274–290 | тест: без `onstart` поля в payload нет (None и пустая строка) |
| `Generate/tests/test_orchestrator.py` | 1280–1437 | тесты пилота: проба на 18188, вердикты, ключ `ports`, пустой и непустой `onstart`, пилотный env |

Входы `docker_image` и `pilot_start_cmd` в `pilot-comfyui.yml` уже имели значение по умолчанию `""` (строки 13 и 17), поэтому их значения не менялись. Блок `env` шага `Run pilot` не менялся: `HF_TOKEN`, `GEN_GEMINI_API_KEY`, `GH_TOKEN` и `WORKER_TOKEN` в пилот не передаются.

Пилотный env после изменений: `GEN_LOG_LEVEL` и `-p 18188:18188`. Ключей `COMFYUI_*`, `PORTAL_CONFIG`, `OPEN_BUTTON_*` и `-p 8000:8000` в нём нет. Команда `onstart` попадает в payload только при непустом `--pilot-start-cmd`.

### 7.2. Результаты пробы и маркеры

Проба: `GET /system_stats` по адресу `http://<public_ipaddr>:<внешний порт из ports["18188/tcp"][0].HostPort>`.

| Результат | Условие | Маркер в логе | Итог |
|---|---|---|---|
| `ok` | HTTP 200 | `PILOT-МАРКЕР порт 18188: HTTP 200 через N с после аренды.` | `PILOT-ИТОГ: результат=ok; …`, код 0 |
| `auth_required` | HTTP 401 или 403: ComfyUI отвечает, нужен токен | `PILOT-МАРКЕР порт 18188: HTTP 401 (или 403) через N с после аренды.` | `PILOT-ИТОГ: результат=auth_required; …`, код 0, не провал и не успех |
| `not_ready` | любой другой код или сетевая ошибка до таймаута | `PILOT-МАРКЕР порт 18188: HTTP <последний код или нет ответа> через N с после аренды.` (при таймауте) | `PILOT-ИТОГ: результат=not_ready; причина=timeout; …`, код 3 |

Остальные маркеры: `PILOT-МАРКЕР running: карта <id> через N с после аренды.`; при отсутствии ключа `18188/tcp` в `ports` — `PILOT: в поле ports нет ключа 18188/tcp. Пришедшие ключи: <имена>.` (ERROR, значения не выводятся). Строка итога содержит `аренда→running=N с; аренда→ответ=N с; карт арендовано N, уничтожено N; уборка выполнена`.

Причина `причина=…` печатается только для `not_ready`: `timeout`, `budget`, `stop`, `lifetime`, `card_stopped`. Решение, принятое без подтверждения: `auth_required` выдаёт код 0, потому что ComfyUI отвечает и инфраструктура работает. Если нужно считать его ошибкой, это одна строка в `_run_pilot`.

### 7.3. UNVERIFIED (проверяется только живым запуском)

1. Формат ключа `18188/tcp` в поле `ports` ответа Vast. В коде ожидается тот же вид, что у `8000/tcp`: список из словаря с `HostPort`.
2. Ответ `/system_stats` без токена: код 200, 401 или 403. Образ может требовать токен по-другому.
3. Поведение Vast без `onstart`: стартует ли образ сам, и через сколько.
4. Ключ `-p 18188:18188` в env для Vast: порт должен открыться в `ports`. Формат передачи `-p` взят из пилотного кода прежней версии и не проверен на этом образе.
5. Доступность `public_ipaddr` и внешнего порта с раннера GitHub.

### 7.4. Риски для generate.yml и как проверить без запуска

Общий код боевого пути менялся только в местах, которые при `pilot=False` не выполняются: `_run_pilot`, `_pilot_verdict`, `_port_keys_text`, нормализация `pilot_start_cmd`. Строка `onstart=self.pilot_start_cmd if self.pilot else None` в `_rent_one` (строка 826) не менялась. Для боевого пути `pilot_start_cmd` всегда `None`, поэтому `onstart` не попадает в payload.

Проверка без запуска:
- `diff -u <архив исходных файлов>/Generate/core/orchestrator.py Generate/core/orchestrator.py`: в диффе не должно быть строк внутри `_rent_one` (774–826), кроме строки 805, в которой меняется только значение константы.
- `grep -n "onstart" Generate/core/orchestrator.py Generate/core/vast_client.py`: `create_instance` должен содержать `if onstart:`.
- Тесты `test_create_instance_without_onstart_has_no_onstart_key` и `test_pilot_empty_start_cmd_sends_no_onstart`: при пустой команде поля `onstart` нет в payload.
- Тест `test_pilot_env_has_no_worker_hf_gemini_gh_or_comfy_keys`: в пилотном env нет `WORKER_TOKEN` и `HF_TOKEN`.
- Тест `test_pilot_probe_goes_to_external_port_of_18188`: проба идёт на внешний порт 18188.
- `grep -rn "PILOT_START_CMD" Generate/ --include=*.py` должен вернуть пустой результат: старая константа удалена из кода (упоминания в разделах 1–5 отчёта исторические).
- Тесты не запускались. Для проверки синтаксиса выполнен только `py_compile` на изменённых файлах.

