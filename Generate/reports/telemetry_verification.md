# Проверка допущений: телеметрия воркера и вывод ComfyUI

**Дата:** 13.10.2026. **Область:** `Generate/worker/` (телеметрия, вывод ComfyUI). Файлы этого документа — проверочный журнал; итоговый отчёт — `telemetry_report.md`.

## Условия проверки

- Живых вызовов Vast, Gemini, Hugging Face и GHCR не было. Коммитов и пушей не было. Образы не собирались.
- **Политика сети (не обходилась):**
  - `api.github.com` и `github.com`: 403, «GitHub access to this repository is not enabled for this session». `add_repo` не вызывался, доступ к репозиторию сессии не расширялся. Из-за этого SHA коммита ComfyUI не записан.
  - `download.pytorch.org` (индекс CPU-колёс): 403 от прокси. Вместо CPU-колеса установлено колесо `torch==2.8.0` с PyPI (`--no-deps`). Файлы `torch/cuda/*.py` в них те же, но это отклонение от брифа.
  - `raw.githubusercontent.com`, `pypi.org`, `files.pythonhosted.org`: доступны.
- Песочница: Linux, 2 vCPU, 8.2 GB RAM, cgroup v1 (лимит памяти не задан), **без GPU, без `nvidia-smi`, без `libnvidia-ml`**. CUDA-библиотеки torch не установлены, поэтому ни один вызов CUDA здесь не выполнялся.
- Python 3.12.3 в venv вне репозитория. Версии: torch 2.8.0, psutil 7.2.2, nvidia-ml-py 13.615.71, pynvml 13.0.1, httpx 0.28.1, pytest 9.1.1.

Обозначения: **VERIFIED** — проверено по исходнику, измерением или тестом, в указанном объёме. **REFUTED** — допущение брифа неверно. **UNVERIFIED** — проверить здесь нельзя; указано, как проверить на карте одним read-only вызовом.

---

## Таблица T1–T10

| № | Утверждение | Как проверено | Результат | Следствие для дизайна |
|---|---|---|---|---|
| T1 | `torch.cuda.mem_get_info()` возвращает `(свободно, всего)` в байтах и работает в контейнере с GPU. | torch 2.8.0, `torch/cuda/memory.py:822–838`: `return torch.cuda.cudart().cudaMemGetInfo(device)`, докстринг «global free and total GPU memory». Запуск невозможен (нет CUDA). | **VERIFIED** по исходнику (порядок и единицы). Работа в контейнере — **UNVERIFIED**. | Воркер эту функцию не вызывает. Вызов в процессе воркера создал бы второй CUDA-контекст на карте; стоимость не измерена. Те же значения берутся из `GET /system_stats` ComfyUI: `comfy/model_management.py` (`get_free_memory`, `get_total_memory`) вызывает `torch.cuda.mem_get_info` в процессе ComfyUI, `server.py:773–782` отдаёт результат. |
| T2 | `torch.cuda.max_memory_allocated()` и `reset_peak_memory_stats()` дают пик на клип, который можно читать и сбрасывать. | `torch/cuda/memory.py:560`: `max_memory_allocated` = `memory_stats()["allocated_bytes.all.peak"]`; `:337` `memory_stats` = `torch._C._cuda_memoryStats(device)`; `:376` сброс = `torch._C._cuda_resetPeakMemoryStats`. Это статистика кэширующего аллокатора **текущего процесса**. | **REFUTED** для процесса воркера. Воркер не выделяет память на GPU. Аллокации ComfyUI идут в другом процессе, их пик из воркера недоступен. В `/system_stats` ComfyUI поля пика нет (`server.py:785–803`). | Поле «пик клипа» не выводится. Варианты: (а) хук внутри процесса ComfyUI (отдельный узел), который читает пик и сбрасывает его на каждый клип — единственный путь к пику аллокатора; (б) опрос NVML «used» устройства во время задачи с частотой не ниже 1 Гц. Выбор за владельцем проекта. |
| T3 | `memory_allocated()` и `memory_reserved()` отличны от занятой памяти драйвера; что писать в пик. | Процессные величины: `torch/cuda/memory.py:539` (`allocated_bytes.all.current`), `:575` (`reserved_bytes.all.current`). Занятость драйвера по устройству: `torch.cuda.device_memory_used` → `nvmlDeviceGetMemoryInfo(...).used` (`torch/cuda/__init__.py`, около строки 1300–1317). Формулы ComfyUI: `vram_free = mem_free_cuda + (reserved − active)`, `torch_vram_free = reserved − active`, `torch_vram_total = reserved` (`model_management.py`, `get_free_memory`, `get_total_memory`). | **VERIFIED** (формулы по исходникам; код `gpu_memory_values` проверен тестами на данных той же формы). Численные значения на карте — **UNVERIFIED**. | В снимке: `видеопамять_занято = vram_total − (vram_free − torch_vram_free)`, `PyTorch_выделено = torch_vram_total − torch_vram_free`, `PyTorch_зарезервировано = torch_vram_total`. Поле «занято» — занятость **драйвера**: она включает CUDA-контекст, рабочие буферы cuBLAS и cuDNN и чужие процессы, и именно она приводит к OOM. Allocated и reserved — диагностика аллокатора ComfyUI, выводятся оба. |
| T4 | Загрузка GPU в процентах недоступна из torch; нужен `nvidia-smi` или `pynvml`. | `torch.cuda.utilization()` (`torch/cuda/__init__.py`, около строк 1347–1361) читает `nvmlDeviceGetUtilizationRates(...).gpu` через pynvml. `_get_pynvml_handler` (строка 1202) бросает `ModuleNotFoundError`, если pynvml не установлен; метаданные torch 2.8.0 пакет pynvml не требуют. В песочнице: `nvidia-smi` отсутствует (`which`), `libnvidia-ml` отсутствует. `pynvml` 13.0.1 — устаревший пакет (FutureWarning при импорте), зависит от `nvidia-ml-py 13.615.71`; `nvmlDeviceGetUtilizationRates` есть; `nvmlInit` без драйвера бросает `NVMLError_LibraryNotFound`. `/system_stats` ComfyUI утилизацию не отдаёт (поиска по `server.py` — ноль совпадений). | torch **без pynvml** источника не даёт: **VERIFIED**. Источник на карте — **UNVERIFIED**: NVML внутри контейнера зависит от того, что драйвер NVIDIA смонтирован в контейнер. | Поле «загрузка GPU» не выводится. Рекомендация: `nvidia-ml-py` с закреплённой версией, вызов напрямую из воркера (без импорта torch). `nvidia-smi` не брать: это отдельный процесс на каждый вызов. Стоимость вызова на GPU не измерена. Проверка на карте: см. раздел «Проверки на карте». |
| T5 | `psutil` есть в образе `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`; если нет, читать `/proc`. | Образ не скачан (правило 1): его содержимое **UNVERIFIED**. Метаданные torch 2.8.0 (установленный dist-info): `psutil` в `Requires-Dist` нет. `requirements.txt` ComfyUI (raw, master, 13.10.2026): строка `psutil` без версии. `comfy/model_management.py:20` и `comfy/system_memory.py:4` импортируют psutil на уровне модуля. `Dockerfile:15` ставит `requirements.txt` ComfyUI. | **REFUTED** как гарантия. psutil не гарантирован ни torch, ни образом (образ не проверен). Он попадает в образ только через `requirements.txt` ComfyUI, без закреплённой версии. | Выбран путь без сторонних пакетов: `/proc/meminfo`, `/proc/stat`, `os.statvfs`, файлы cgroup. psutil в воркере не используется. |
| T6 | Чтение `/proc/stat` требует двух выборок; формула и поведение первой выборки. | `psutil/_pslinux.py:501–513` (`cpu_times` читает первую строку `/proc/stat`). `psutil/__init__.py:1733–1760`: `_cpu_tot_time` (сумма, для Linux гостевые поля не суммируются), `_cpu_busy_time` = total − idle − iowait. Измерено: `psutil.cpu_percent(interval=None)`: первый вызов = **0.0**, второй = **2.6** через 0.2 с. | **VERIFIED**. | Формула: `100·(Δвсего − Δпростой)/Δвсего` по полям user…steal; простой = idle + iowait. Первая выборка выводится как `ЦП=н/д`, а не 0.0: 0.0 — выдуманное значение. Тики `/proc/stat` — 1/USER_HZ (обычно 10 мс), интервал короче тика даёт `н/д` (наблюдалось в тестах). В контейнере `/proc/stat` может показывать хост: **UNVERIFIED** на карте. |
| T7 | `shutil.disk_usage` даёт файловую систему контейнера; размер 100 GB подтверждается. | `shutil.disk_usage().free` равен `f_bavail × f_frsize` из `os.statvfs` (измерено: `True`). В песочнице корень: 270.55 GB всего, 31.87 GB доступно (`df -B1`). `disk_gb: 100` (`config.default.yaml:79`) передаётся в Vast как `disk` (`vast_client.py:412`). Что видит процесс в контейнере Vast, зависит от storage-driver хоста: проверить невозможно. | Семантика — **VERIFIED**. Размер 100 GB — **UNVERIFIED** (не подтверждается изнутри контейнера). | Поле «диск_свободно» пишется как есть: `statvfs` по пути вывода (`COMFY_OUTPUT_DIR`, по умолчанию `/workspace/ComfyUI/output`, тот же каталог `/workspace`, где модели). Значение «100 GB» в снимок не подставляется. |
| T8 | ComfyUI печатает сообщения загрузки и выгрузки моделей в stdout; точные строки, файл и строка. | Исходники ComfyUI master (raw, 13.10.2026). **SHA коммита не записан** (GitHub закрыт политикой). Хеши blob (`git hash-object`) для сверки с деревом коммита: `main.py` b93c2a81…, `server.py` ca0399df…, `execution.py` d4a68b32…, `app/logger.py` fe82c40c…, `comfy/model_management.py` 13b690bd…, `comfy/model_patcher.py` 8472b4a4…. Таблица строк ниже. | Строки — **VERIFIED** для master на дату проверки. Утверждение «stdout» — **REFUTED**: по умолчанию сообщения идут в **stderr** (`app/logger.py:120`, `StreamHandler()` без аргумента; `comfy/cli_args.py:225`: `--log-stdout` выключен по умолчанию). Версия в образе — **UNVERIFIED** (не закреплена, `Dockerfile:14–15`). | Захватывать оба потока (`2>&1`). Сообщения «Unloading …» и «freed …» — уровень **DEBUG** (`model_management.py:920`, `model_patcher.py:1263`), при INFO не видны. Поэтому `setup.sh` передаёт `--verbose DEBUG` только при `GEN_LOG_LEVEL=DEBUG`. |
| T9 | Stdout ComfyUI не захватывается буфером воркера; способ захвата; буферизация и `PYTHONUNBUFFERED`. | `setup.sh:103` (до изменения) запускал ComfyUI фоновым процессом без перенаправления; `server.py` пишет в буфер только через `logging` (обработчик на `Generate.worker`) и печатает одну строку сборки. Вывод ComfyUI в буфер не попадал: **VERIFIED**. ComfyUI заменяет `sys.stdout` и `sys.stderr` на `LogInterceptor` (`app/logger.py:107–108`), у которого `TextIOWrapper` создан без `write_through`. **Реплика** этих классов (`t9/child.py`, без зависимостей): строки `logging` приходят через **0.04 с** (flush в `StreamHandler`), строка `print()` — только при выходе процесса, через **2.06 с**. Одинаково при `PYTHONUNBUFFERED=1` и без него (`Dockerfile:19` задаёт `1`, на `print()` не влияет). ANSI-коды есть всегда (`ColoredFormatter`, `logger.py:39`). | Захват: **VERIFIED** на реплике классов, **UNVERIFIED** на настоящем бинарнике ComfyUI. Задержка `print()`: **UNVERIFIED** на карте; исправление требует правки ComfyUI и не входит в эту задачу. | Выбран файл + `tee` + чтение воркером (обоснование в отчёте). Проверка bash: `$!` указывает на PID python, файл получает вывод, stdout контейнера тоже получает его (проверено скриптом-заглушкой). |
| T10 | Один вызов `/proc` и torch дёшев; раз в секунду `nvidia-smi` — нет. | Измерено 60 вызовов каждого (таблица в разделе «Измерения»). `nvidia-smi` в песочнице нет, измерить его нельзя. Нижняя граница: fork+exec `/bin/true` = 1.49 мс (среднее). | `/proc` и `statvfs` — **VERIFIED** измерением. torch — **не измерен** (нет CUDA). `nvidia-smi` — **не измерен**; величина затрат **UNVERIFIED**, но нижняя граница уже 1.5 мс на вызов. | `nvidia-smi` в дизайне не используется. Опрос раз в секунду даже одним spawn-ом съедал бы около 0.15 % одного vCPU на этой машине; реальный `nvidia-smi` дороже, но это не измерено. |

---

## Дополнительные допущения (найдены по ходу)

| № | Утверждение | Проверка | Результат | Следствие |
|---|---|---|---|---|
| E1 | `/system_stats` отдаёт по каждому устройству `vram_total`, `vram_free`, `torch_vram_total`, `torch_vram_free`. | `server.py:773–782` (исходник master). Воркер уже читает `devices[0].vram_free` в `/health` (`server.py:626–637`). | **VERIFIED** по исходнику. Значения на карте — **UNVERIFIED**. | Основной источник видеопамяти в снимке. Если поля нет в версии ComfyUI из образа, поле пропускается с предупреждением, без выдуманных значений. |
| E2 | `Dockerfile` копирует файлы воркера явным списком. | `Dockerfile:34` до изменения: перечислены только `server.py`, `comfy_client.py`, `setup.sh`. | **VERIFIED**. Без правки образ не получил бы новые модули и не запустился бы. | Добавлены `comfy_output.py` и `telemetry.py`. Тест `test_dockerfile_copies_every_worker_module` не даёт забыть новый модуль. |
| E3 | Оркестратор менять не нужно: он сам выгружает и печатает строки ComfyUI. | `orchestrator.py:305–333`: печатаются все строки после курсора, с префиксом `[карта N]`, текст проходит `_mask_remote`. | **VERIFIED**. | Изменений в `orchestrator.py` нет. Побочный эффект: строки ComfyUI попадают в лог запуска и в `card_<слот>.log` (SPEC, раздел 13: репозиторий публичный). См. риски. |
| E4 | ComfyUI учитывает предел памяти cgroup. | `comfy/system_memory.py:113–114` (`cgroup_memory_limit`), `_limit_in` (`memory.max`, затем `memory.limit_in_bytes`). В песочнице: `memory.max` нет; `memory.limit_in_bytes` = `9223372036854771712` (лимита нет). | **VERIFIED**. | Воркер читает тот же файл. Поле «ОЗУ_лимит_контейнера» выводится, только если предел меньше `MemTotal`. |
| E5 | `/proc/meminfo` и `/proc/stat` в контейнере отражают ресурсы хоста. | Стандартное поведение Docker без LXCFS. В песочнице (ВМ) не проверяется. | **UNVERIFIED** на карте. | Поле «ОЗУ_доступно» может показывать память хоста, а не контейнера; при заданном лимите cgroup это видно по полю «ОЗУ_лимит_контейнера». Проверка на карте: см. раздел «Проверки на карте». |
| E6 | `pynvml` устарел; вместо него `nvidia-ml-py`. | FutureWarning при импорте `pynvml` (измерено); `pynvml` требует `nvidia-ml-py` (метаданные). | **VERIFIED**. | В рекомендации указан `nvidia-ml-py`. |
| E7 | ComfyUI печатает сообщения загрузки на уровне INFO без дополнительных флагов. | `main.py:22`: `get_console_log_level(args.verbose)`; при пустом `--verbose` возвращается `'INFO'` (`cli_args.py:52–54`). | **VERIFIED**. | Строки «Requested to load …», «loaded completely …», «… models unloaded.», «Unloaded partially …» видны без флагов. |
| E8 | Конструкция `"${COMFY_VERBOSE_ARGS[@]}"` при пустом массиве и `set -u` требует bash 4.4 и новее. | Проверено в bash песочницы (5.x). | **VERIFIED** в песочнице. Версия bash в базовом образе — **UNVERIFIED**. | Базовый образ — Ubuntu-based; для bash ≥ 4.4 проверить на карте (`bash --version`). |
| E9 | Индекс CPU-колёс PyTorch доступен. | `download.pytorch.org`: 403 от прокси. | **REFUTED** (политика сети). | Использовано PyPI-колесо `torch==2.8.0` с `--no-deps` только для чтения исходников. Отклонение от брифа зафиксировано. |
| E10 | `> >(tee …)` сохраняет `$!` равным PID процесса ComfyUI. | Тест bash: `ps -o args= -p $!` показывает python. | **VERIFIED**. | `setup.sh` сохраняет `COMFY_PID=$!` и далее использует его без изменений. |
| E11 | Строки ComfyUI не содержат текст промпта. | Не проверяется: ComfyUI при ошибке может напечатать входные значения (трассировка, `Failed to validate prompt`, `execution.py:638–639`, `1233–1257`). | **UNVERIFIED**. | Риск по SPEC 0.1 (промпт в лог не пишем). Проверить на первом реальном прогоне: поиск фрагментов промптов в логах карты. |
| E12 | Имена полей `/system_stats` и строки логов ComfyUI совпадают с версией в образе. | Образ не закреплён и не собран; SHA коммита неизвестен. Проверено по master на дату. | **UNVERIFIED** для образа. | Если поля отличаются, поле снимка пропускается с одним предупреждением; строки логов ComfyUI выводятся как есть, без разбора. |

---

## Строки сообщений ComfyUI (T8)

Все строки — из `logging` (поток stderr по умолчанию). Файлы и строки — по исходнику master на дату проверки.

| Строка (дословно, английский) | Файл:строка | Функция | Уровень | Видна при INFO |
|---|---|---|---|---|
| `Total VRAM {…} MB, total RAM {…} MB` | `comfy/model_management.py:366` | на уровне модуля (при импорте) | INFO | да |
| `RAM limited by cgroup to {…} MB (host has {…} MB)` | `comfy/model_management.py:369` | на уровне модуля | INFO | да, если есть лимит |
| `Set vram state to: {…}` | `comfy/model_management.py:597` | на уровне модуля | INFO | да |
| `Unloading {ClassName}` | `comfy/model_management.py:920` | `free_memory` (стр. 893) | **DEBUG** | нет |
| `Requested to load {ClassName}` | `comfy/model_management.py:978` | `load_models_gpu` (стр. 939) | INFO | да |
| `{N} models unloaded.` | `comfy/model_management.py:1011` | `load_models_gpu` | INFO | да |
| `loaded partially; … MB loaded, … MB offloaded, …` | `comfy/model_patcher.py:1113` | `ModelPatcher.load` (стр. 1002) | INFO | да |
| `loaded completely; … MB loaded, full load: …` | `comfy/model_patcher.py:1116` | `ModelPatcher.load` | INFO | да |
| `freed {n}` | `comfy/model_patcher.py:1263` | `ModelPatcher.partially_unload` (стр. 1191) | **DEBUG** | нет |
| `Unloaded partially: … MB freed, … MB remains loaded, …` | `comfy/model_patcher.py:1273` | `ModelPatcher.partially_unload` | INFO | да |
| `Prompt executed in {…} seconds` | `main.py:398` (вариант без секунд: `main.py:396`) | `prompt_worker` (стр. 325) | INFO | да |
| `Memory summary:` + сводка | `execution.py:644` | обработка OOM | INFO | да |
| `Got an OOM, unloading all loaded models.` | `execution.py:645` | обработка OOM | ERROR | да |

**Освобождение памяти без собственного сообщения:** `soft_empty_cache` (`model_management.py:2131`) и `unload_all_models` (`:2149`) строк логирования не содержат (проверено поиском в диапазоне 2131–2178). Освобождение видно через «N models unloaded.» и «Unloaded partially …».

---

## Измерения (T10), песочница, 60 вызовов каждого

| Операция | Среднее | Медиана | Суммарно на 60 вызовов |
|---|---|---|---|
| A. чтение и разбор `/proc/meminfo` | 19.8 мкс | 16.6 мкс | 1.19 мс |
| B. чтение и разбор первой строки `/proc/stat` | 9.0 мкс | 8.7 мкс | 0.54 мс |
| C. `shutil.disk_usage('/')` | 1.3 мкс | 1.2 мкс | 0.08 мс |
| D. HTTP GET к локальной заглушке `/system_stats` (keep-alive, JSON) | 1760.9 мкс | 1646.0 мкс | 105.7 мс |
| E. нижняя граница для любого `nvidia-smi`: fork+exec `/bin/true` | 1486.3 мкс | 1435.8 мкс | 89.2 мс |
| F. `nvidia-smi` | не измерено (нет в системе) | — | — |
| G. полный снимок `SystemSampler.snapshot` (GPU-заглушка, реальные `/proc` и `statvfs`) | 102.5 мкс | 88.6 мкс | — |
| H. `torch.cuda.*` | не измерено (нет CUDA) | — | — |

**Выводы:**
- Полный снимок стоит около 0.1 мс. Раз в минуту и раз в 10 с это пренебрежимо мало.
- D измерен на localhost и заглушке, не на ComfyUI. Реальная стоимость `/system_stats` на карте — **UNVERIFIED**.
- Раз в секунду E даёт около 1.5 мс в секунду, то есть около 0.15 % одного vCPU на этой машине, и это только порождение процесса. Затраты самого `nvidia-smi` (инициализация NVML) не измерены.

---

## Проверки на карте (одним read-only вызовом каждая)

1. **Видеопамять и поля ComfyUI:** `curl -s -H "X-Worker-Token: …" http://127.0.0.1:8000/health` и `curl -s http://127.0.0.1:8188/system_stats | python -m json.tool | grep -E "vram|torch_vram"`. Ожидается четыре поля `*vram*` для первого устройства.
2. **Версия ComfyUI в образе:** `git -C /workspace/ComfyUI rev-parse HEAD` (Dockerfile клонирует репозиторий `git clone`, поэтому `.git` есть) и `curl -s http://127.0.0.1:8188/system_stats`, поле `system.comfyui_version` (`server.py`, блок `system_stats`). Нужно для закрепления версии (SPEC, раздел 13).
3. **Драйвер NVML и загрузка GPU:** `python -c "import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)"`. Если падает с `NVMLError_LibraryNotFound`, драйвер в контейнер не смонтирован и поле загрузки остаётся недоступным.
4. **Память хоста против контейнера:** `grep MemTotal /proc/meminfo; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes`. Если лимит меньше `MemTotal`, поле «ОЗУ_доступно» завышено относительно контейнера.
5. **Диск:** `df -B1 /workspace` и сравнение с `disk_gb` (100). Если размер не 100 GB, в снимке видно реальное значение.
6. **Версия bash:** `bash --version` в образе (нужно 4.4 и новее для `"${COMFY_VERBOSE_ARGS[@]}"`).
7. **Задержка `print()` в ComfyUI:** на реальном прогоне сравнить время появления строк `print()` в `/workspace/comfyui.log` с временем их вывода. Ожидается задержка до выхода или до 8 КиБ.
8. **Утечка промптов в строках ComfyUI:** `grep -c` фрагментов текста промпта в `/workspace/comfyui.log` после первого клипа. Ненулевой результат — отдельная задача по маскированию.
