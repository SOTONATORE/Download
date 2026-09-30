#!/usr/bin/env python3
"""
search.py

Недостающее звено между generate_queries.py и download.py: читает requests.json
(сгенерированный generate_queries.py) и для каждого сегмента подбирает
подходящее стоковое/архивное фото или видео на pexels/pixabay/wikimedia/nasa/loc,
используя лицензионный фильтр, текстовый фильтр по сущностям и CLIP-ранжирование
по схожести превью с текстом запроса.

На выходе:
    links.txt        - строки "номер: URL" в формате, который уже понимает
                        download.py (передаётся туда как переменная окружения
                        INPUT_LINKS)
    backup_links.txt - тот же формат "номер: URL", но резервная ссылка на ДРУГОЙ
                        файл для того же сегмента (передаётся в download.py как
                        INPUT_BACKUP_LINKS) - используется, если скачивание
                        primary-ссылки не удалось из-за транзиентного сбоя
                        соединения. Покрывает не все номера из links.txt - для
                        части сегментов backup просто не находится, это
                        нормальный случай (см. try_claim_backup)
    missing.txt      - номера сегментов, для которых ничего подходящего не нашлось

Использование:
    python search.py --input requests.json --links-output links.txt \
        --backup-links-output backup_links.txt --missing-output missing.txt

Переменные окружения:
    PEXELS_API_KEY, PIXABAY_API_KEY   - обязательны (в т.ч. для broad-варианта каскада)
    SEARCH_SEM_PEXELS / _PIXABAY / _WIKIMEDIA / _NASA / _LOC
        - per-site семафоры одновременных запросов (умолч. 5/5/10/10/10)
    SEARCH_LOC_MIN_INTERVAL_SECONDS
        - глобальный (на весь запуск, не per-задача) минимальный интервал между
          ПОСЛЕДОВАТЕЛЬНЫМИ запросами к LOC, вне зависимости от того, сколько
          сегментов обрабатывается параллельно. Это отдельный механизм от
          SEARCH_SEM_LOC: семафор ограничивает только конкурентность (сколько
          запросов летит ОДНОВРЕМЕННО), а не частоту (сколько запросов в секунду
          в принципе уходит) - при высокой глобальной параллельности сегментов
          семафор сам по себе не мешает 10 запросам уйти почти синхронно, а затем
          ещё 10 через долю секунды, и LOC отвечает 429 почти на всё подряд.
          Официальный лимит LOC (https://www.loc.gov/apis/json-and-yaml/working-within-limits/)
          - 20 запросов/мин у JSON/YAML API, при превышении - блокировка IP на 1 ЧАС
          (не на минуту!). Умолч. 3.2 сек (~18.75 запросов/мин, ~6% запас от потолка -
          пересчитано по итогам реального прогона на 126 сегментах, где именно этот
          интервал оказался доминирующим узким местом всего скрипта: 104 из 126
          сегментов трогали LOC, а при старом значении 5.0с/~12 запросов/мин, ~40%
          запас, это одно давало ~520с из ~620с общего времени прогона). 3.2с - намеренно
          НЕ математически ровно 5% запаса (=3.158с), а округлено В БОЛЬШУЮ сторону
          (~6.25% запаса) - чтобы дрожание таймингов asyncio/event loop не утащило
          фактическую частоту выше 19 запросов/мин и не спровоцировало тот самый часовой
          бан, который вся эта конструкция должна предотвращать. Если 429 всё равно
          появляются - увеличивайте (сначала обратно к 5с, если совсем плохо - выше).
    SEARCH_LOC_PREVIEW_MIN_INTERVAL_SECONDS
        - лимитер для скачивания превью LOC (tile.loc.gov, лимит 150/мин). Умолч. 0.45с.
    SEARCH_WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS
        - глобальный лимитер между последовательными запросами превью Wikimedia Commons
          (upload.wikimedia.org) для защиты от 429 при высокой конкурентности (умолч. 0.35с).
    SEARCH_WIKIMEDIA_PREVIEW_CONCURRENCY
        - семафор одновременных загрузок превью Wikimedia Commons (умолч. 3).
    SEARCH_LOC_TIMEOUT_SECONDS
        - таймаут поискового запроса к LOC (умолч. 30с; увеличено с 15с, чтобы устранить
          постоянные таймауты на 1-й попытке из-за медленного холодного поиска LOC).
    SEARCH_GLOBAL_CONCURRENCY  - сколько сегментов обрабатывать параллельно (умолч. 40)
    SEARCH_CLIP_CONCURRENCY   - сколько CLIP-инференсов одновременно (умолч. 2, CPU-bound)
    SEARCH_CANDIDATES_PER_SITE - сколько топ-кандидатов с сайта пускать под CLIP (умолч. 5)
    SEARCH_CLIP_MODEL / SEARCH_CLIP_PRETRAINED - модель open_clip (умолч. ViT-B-32-quickgelu / openai)
    SEARCH_SIM_MIN_THRESHOLD   - нижняя страховка: кандидат с own_sim ниже не принимается
        ни по какому критерию (умолч. 0.21 - см. "Шестое уточнение" в докстринге ниже)
    SEARCH_SIM_ACCEPT_THRESHOLD - абсолютный критерий принятия (умолч. 0.30 - см. там же)
    SEARCH_REL_TOP_N (умолч. 10 - потолок), SEARCH_REL_MARGIN (0.03), SEARCH_FLAT_SPREAD (0.03)
        - относительная оценка, см. раздел "Относительная оценка CLIP" ниже

Возвращаемые коды:
    0 - links.txt и missing.txt успешно записаны (даже если часть/все сегменты в missing)
    1 - структурная ошибка (невалидный API-ключ, битый requests.json и т.п.)

Зависимости:
    pip install aiohttp pillow open_clip_torch
    (torch лучше ставить отдельно, CPU-only wheel, см. комментарий в конце файла)

ВАЖНОЕ ДОПУЩЕНИЕ по интерпретации п.6-7 исходного ТЗ (спецификация была неоднозначна
в этом месте): "лучший кандидат сайта >= порога accept" останавливает перебор сайтов, но
выбор и дедуп-бронирование всегда идёт по НАКОПЛЕННОМУ пулу кандидатов (>= порога min) со
всех уже пройденных сайтов, отсортированному по убыванию similarity - а не только по
кандидатам текущего (последнего) сайта. Это единственное прочтение, совместимое одновременно
с "не пробовать остальные сайты" (п.6) и "среди кандидатов ПО ВСЕМ ПЕРЕБРАННЫМ САЙТАМ" (п.7).
Если имелось в виду не так - легко поменять в try_claim_pool/process_segment.

Второе допущение: правило "нет превью для CLIP -> дисквалифицировать именно этого
кандидата, не весь сайт" (изначально уточнено для видео на Wikimedia) применено ко ВСЕМ
сайтам единообразно в fetch_preview_bytes/score_candidates - это строго безопаснее и
не создаёт особых случаев.

Третье допущение (rate-limiting LOC): семафор SEARCH_SEM_LOC и rate-limiter
SEARCH_LOC_MIN_INTERVAL_SECONDS решают РАЗНЫЕ задачи и работают одновременно -
семафор по-прежнему ограничивает, сколько запросов к LOC могут физически висеть
в полёте одновременно, а rate-limiter поверх этого гарантирует минимальный зазор
по времени между началом двух последовательных запросов (глобально по всему
запуску, а не per-сегмент/per-задача).

Четвёртое (важное) уточнение по LOC, добавленное после реального прогона: по
официальной документации LOC (working-within-limits) превышение лимита JSON/YAML
API (20 запросов/мин) приводит к блокировке IP на ЦЕЛЫЙ ЧАС, а не к обычному
кратковременному 429. Это значит, что как только LOC один раз ответил 429,
дальнейшие ретраи с exponential backoff (секунды-десятки секунд) внутри ТЕКУЩЕГО
запуска бессмысленны - блокировка всё равно не снимется за время работы CI-джобы.
Поэтому search_loc теперь вызывает http_get_json с treat_429_as_exhaustion=True
(как pexels/pixabay) - первый же 429 сразу помечает "loc" исчерпанным на весь
остаток запуска, вместо повторных попыток достучаться до сайта, который уже точно
не ответит. Сегменты, где loc стоит не последним в sites, просто продолжают перебор
остальных сайтов - это поведение уже было и не менялось.

Также LOC может отдать вместо JSON html-страницу с CAPTCHA при перегрузке на своей
стороне (см. ту же страницу документации: "users may encounter ... HTML pages with
CAPTCHAs even when operating below the rates listed above") - это ловится отдельно
как aiohttp.ContentTypeError при попытке resp.json() и обрабатывается так же, как
429 (тот же treat_429_as_exhaustion), т.к. по сути это тот же сигнал "нас блокируют".

Пятое (критичное) уточнение - баг конфигурации CLIP, найденный после прогона с
0 найденных из 126 сегментов СРАЗУ ПО ВСЕМ сайтам (не только loc): модель бралась
как SEARCH_CLIP_MODEL=ViT-B-32 (без суффикса) с pretrained=openai. Это известный
баг open_clip (https://github.com/mlfoundations/open_clip/issues/771): чекпоинт
"openai" для B/32 обучен с QuickGELU-активацией, но конфиг архитектуры "ViT-B-32"
(без суффикса) по умолчанию использует обычный GELU - в логах это видно как warning
"QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag
'openai' (quick_gelu=True)". Из-за этого модель технически загружается и работает
без ошибок, но выдаёт бессмысленные эмбеддинги - и КАЖДЫЙ кандидат на КАЖДОМ сайте
получает around-случайный/заниженный similarity, падающий ниже порога. Это системная
причина сразу для всех сайтов одновременно, не связанная с лицензиями, запросами или
сущностями. Исправлено: дефолт SEARCH_CLIP_MODEL сменён на ViT-B-32-quickgelu (тот же
pretrained=openai, но с правильной активацией).

Шестое уточнение (по итогам прогона на 126 сегментах после фикса QuickGELU) - пороги
similarity были подобраны "на глаз" и оказались нереалистично высокими для СЫРОГО (без
температурного скейлинга/софтмакса) косинусного сходства CLIP: у настоящих релевантных
пар текст-картинка raw cosine similarity типично лежит в диапазоне ~0.2-0.35, а не
0.5-0.85, как было выставлено изначально. Сводка по прогону это подтвердила напрямую:
pexels/loc присылали в CLIP сотни нормальных кандидатов с avg similarity 0.20-0.28 и
best 0.33-0.36 - это здоровые значения для настоящих совпадений, просто ниже прежнего
порога отсечения 0.5, из-за чего пул почти всегда оказывался пуст. Пороги пересчитаны
под этот диапазон (SIM_MIN_THRESHOLD 0.5->0.21, SIM_ACCEPT_THRESHOLD 0.85->0.30) и
вынесены в переменные окружения SEARCH_SIM_MIN_THRESHOLD / SEARCH_SIM_ACCEPT_THRESHOLD,
чтобы их можно было донастроить по факту (например по перцентилю на своей выборке
сегментов), не трогая код.

Седьмое уточнение - у pixabay и wikimedia в тестовом прогоне 100% превью не скачивались
(fetch_preview_bytes возвращал None для всех кандидатов, 0 ушло в CLIP), при этом сам
поиск (raw/license/keyword) отрабатывал нормально. Причина - типичная защита CDN от
хотлинкинга: запрос к самому медиафайлу без Referer (иногда и Origin), указывающего на
страницу-источник, отклоняется (403/406 и т.п.), даже если User-Agent в порядке (User-
Agent уже был поправлен раньше для API-запросов к Wikimedia, но не для скачивания самих
превью-картинок). Исправлено: fetch_preview_bytes теперь подставляет Referer (и Origin,
выведенный из него) по каждому сайту - для wikimedia используется page_url конкретного
кандидата (страница файла), если он есть, иначе общий https://commons.wikimedia.org/;
для остальных сайтов - их основной домен. Также при неуспехе (status != 200 или сетевая
ошибка) теперь логируется DEBUG-строка с сайтом, id кандидата, HTTP-статусом и превью
тела ответа - раньше fetch_preview_bytes молча возвращал None без единой детали, что и
не давало отличить блокировку по Referer от любой другой причины.

Восьмое уточнение - text_matches_keywords делал точное вхождение подстроки, из-за чего
разные способы транслитерации одного и того же имени (например "Abdulmecid" в
entity_keywords против "Abdülmecid" в тексте кандидата, или "Abdul Mejid" против
"Abdulmecid") не совпадали, хотя семантически это один и тот же человек. Не подключая
внешних fuzzy-библиотек, добавлены два дешёвых слоя поверх прежней точной проверки:
(1) нормализация через unicodedata (NFKD + снятие комбинирующих диакритических знаков),
которая сама по себе схлопывает "Abdülmecid" -> "abdulmecid"; (2) до-проверка на уровне
отдельных слов текста через стандартный difflib.SequenceMatcher (стандартная библиотека,
без новых зависимостей) с порогом схожести - ловит близкие, но не идентичные варианты
написания вроде "Mejid"/"Mecid". Это осознанно не полноценный fuzzy-matching (без
rapidfuzz и т.п.) - как и просили, достаточно "чего-то попроще" поверх точного совпадения.

Девятое уточнение - баг в самой реализации восьмого пункта, найденный по логу реального
прогона: entity_keywords вида "Abdulmecid II" / "Абдул-Меджид II" (многословные, с
римской цифрой и/или дефисом) НИ РАЗУ не проходили фильтр, хотя у LOC на них находилось
всего 1-2 сырых кандидата - т.е. почти наверняка релевантных, отсеянных ошибочно. Причина:
_fuzzy_word_match сравнивал ключевую фразу ЦЕЛИКОМ (например "abdulmecid ii" или
"абдул-меджид ii" одной строкой) с ОТДЕЛЬНЫМИ словами текста кандидата - при такой разнице
в длине difflib.SequenceMatcher.ratio() почти всегда оказывается ниже порога, даже если
имя в тексте присутствует дословно. Дефис в "Абдул-Меджид" дополнительно ломал даже точное
вхождение подстроки (текст мог использовать пробел вместо дефиса или наоборот). Исправлено:
text_matches_keywords теперь токенизирует ключевую фразу ТЕМ ЖЕ способом (_WORD_RE), что и
текст кандидата, и требует, чтобы КАЖДОЕ значимое слово фразы (длиннее _MIN_FUZZY_WORD_LEN)
нашлось в тексте кандидата точно или fuzzy-приближённо - короткие токены (римские цифры,
инициалы) в этой пословной проверке пропускаются и не блокируют совпадение по остальным
словам той же фразы, а не отбрасывают всю фразу целиком, как раньше.

Десятое уточнение - найдено по DEBUG-логу реального прогона (round 4): у ПРЯМОГО pixabay
(media_type="video") preview_url оказывался None у 100% кандидатов (105/105 во всех 21
сегменте, где pixabay был первичным сайтом), при этом pixabay_broad (тогда ещё pixabay_fallback) работал почти
нормально (missing только 15/235). Причина - устаревшее допущение о структуре ответа
Pixabay Video API: код брал hit.get("picture_id") и строил превью через vimeocdn
(https://i.vimeocdn.com/video/{picture_id}_200x150.jpg), но проверка по актуальной
официальной документации (https://pixabay.com/api/docs/) показала, что в СЕГОДНЯШНЕМ
ответе /api/videos/ поля "picture_id" на верхнем уровне хита нет вообще - вместо этого
у каждого видео есть "videos": {large/medium/small/tiny}, и превью-картинка лежит в
videos.<size>.thumbnail. (Кандидаты, на которых это всплыло, по текстам тегов - явно
видео, не фото: "time lapse", "aerial, drone, cinematic", "live wallpaper" и т.п. - это
подтверждает, что 105/105 отказов пришлись именно на видео-ветку, а не на фото.) У фото
(previewURL/webformatURL) структура ответа не менялась и совпадает с текущими доками -
там всё было верно уже раньше. Исправлено: для video берём первый доступный thumbnail
из videos.tiny -> small -> medium -> large (tiny предпочтителен - меньше трафика для
CLIP, где точность превью не критична, важна только similarity с текстом).

Одиннадцатое уточнение (Этап 1 стабилизации):
1. CDN превью Wikimedia Commons (upload.wikimedia.org) банил скрипт по 429 из-за
   одновременных запросов от 40 параллельных сегментов (в логе было 43 отсеянных кандидата).
   Добавлены WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS (0.35с) и WIKIMEDIA_PREVIEW_CONCURRENCY (3),
   а в fetch_preview_bytes добавлена честная обработка 429 с учетом заголовка Retry-After
   и паузой перед попыткой 2 вместо прежнего мгновенного return None.
2. Защита от связки "primary LOC + backup LOC": если primary взят с LOC, try_claim_backup
   больше НЕ берет дубликат с LOC (возвращает None). В find_backup для LOC primary принудительно
   подключаются pexels и pixabay, что позволяет найти независимый кросс-сайтовый backup по
   каскаду запросов (broad идёт на pexels/pixabay) и спасает файл при часовом бане LOC на этапе download.py.
3. Фильтрация URL в search_loc: исключены страницы виртуальных выставок (/exhibits/), блогов
   и порталов, которые возвращают HTML при запросе ?fo=json (баг сегмента 6). Принимаются только
   оцифрованные каталожные объекты (/item/ и /resource/).
4. Обогащение текста LOC: поле subject/subjects включено в text кандидата, что спасает до 44%
   кандидатов от ложного отсева фильтром сущностей (Анкара, Мехмед VI, Сан-Ремо и др.).
5. Поиск Wikimedia Commons дополнен filetype:bitmap для картинок, исключая 500+ сканов PDF/DjVu.
6. Таймаут поиска LOC увеличен до 30 секунд (SEARCH_LOC_TIMEOUT_SECONDS=30) для устранения
   постоянных таймаутов на первой попытке холодного поиска.

Относительная оценка CLIP:
 - Перед поиском scene всех сегментов кодируются ОДИН раз (батчи по 64) в матрицу нормализованных
   векторов. Превью кандидата кодируется один раз (кэш по (site, cand_id), только вектор ~2 КБ);
   сходство со всеми scene - одно матричное умножение. Текст запроса в CLIP больше не кодируется.
 - Принят, если own_sim >= SIM_MIN_THRESHOLD (0.21) И (а) его сегмент в топ-N сходств среди всех
   сегментов, или (б) own_sim >= SIM_ACCEPT_THRESHOLD (0.30). N без явного SEARCH_REL_TOP_N =
   max(3, min(10, ceil(5% сегментов))); явное значение берётся как есть.
 - "Чужой": другая scene выше собственной больше чем на SEARCH_REL_MARGIN (0.03) - не принимается
   по рангу (а); абсолютный критерий (б) остаётся в силе.
 - Однотемные ролики: если разброс сходств кандидата по scene меньше SEARCH_FLAT_SPREAD (0.03),
   ранг неинформативен - решает только (б) и нижняя страховка.
 - Порядок best-effort: сначала кандидаты не "чужие" (по убыванию own_sim), потом "чужие".
 - Для калибровки смотрите строки "Сегмент N: выбран site/id [вариант] own_sim rank причина" и
   итоги "Итог выбора primary/backup" (по рангу / абс. порогу / best-effort по причинам, min/median/max
   own_sim). Пороги 0.21 / 0.30 / margin 0.03 заданы по оценке и подбираются по этому итогу.
 - Режим "лучший из найденного": если по всему каскаду никто не принят, берётся лучший по own_sim
   (ключ статистики best-eff); так же и для backup. Ручной проверки нет.

Каскад запросов (requests.json: query_narrow / query_medium / query_broad):
 - Порядок вариантов: первый сайт сегмента архивный (wikimedia/loc/nasa) -> narrow, medium,
   broad; первый сайт сток (pexels/pixabay) -> medium, narrow, broad. Каскад ленивый:
   следующий вариант запускается, только если предыдущий не дал принятого и
   заклеймленного кандидата (build_cascade - чистая функция, run_variant - оценка варианта).
 - Сайты: при архивном первом сайте narrow/medium идут ТОЛЬКО на архивные сайты из seg.sites
   (имена собственные не уходят на Pexels, лимит 200/час не тратится); при стоковом первом
   сайте - на все сайты из seg.sites; broad всегда только на pixabay и pexels.
 - Pixabay в любом списке сайтов стоит ПЕРЕД Pexels (лимит Pixabay щедрее); дублей нет.
 - Фильтр сущностей (entity_filter_applies): только narrow/medium и только архивные сайты;
   для broad и стоковых сайтов никогда. Исключение skip_keyword_filter (pexels+video) сохранено.
 - Дедупликация: вариант пропускается, если запрос пуст/None или совпадает (без регистра, с
   схлопнутыми пробелами) с уже включённым вариантом на пересекающихся сайтах.
 - Ключи статистики: narrow/medium - имя сайта, broad - "<сайт>_broad", backup - "<сайт>_backup".
 - Backup использует тот же каскад; сайты варианта пересекаются с допустимыми для backup.
 - Проверка без сети: python search.py --selftest
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import functools
import io
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace as dc_replace
from typing import Any, Awaitable, Callable, Optional, Sequence
from urllib.parse import urlparse

from media_formats import (
    VIDEO_EXTENSIONS,
    FormatRejectStats,
    ext_from_name,
    is_allowed_ext,
)

try:
    import aiohttp
except ImportError:
    print("Не найден aiohttp. Установите: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

try:
    import torch
    import open_clip
    from PIL import Image
except ImportError:
    print(
        "Не найдены torch/open_clip/pillow. Установите (CPU-only torch, чтобы не тянуть "
        "CUDA-сборку в CI):\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  pip install open_clip_torch pillow\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Константы и настройки из окружения
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_INPUT = "requests.json"
DEFAULT_LINKS_OUTPUT = "links.txt"
DEFAULT_MISSING_OUTPUT = "missing.txt"
DEFAULT_BACKUP_LINKS_OUTPUT = "backup_links.txt"
DEFAULT_BACKUP_MISSING_OUTPUT = "backup_missing.txt"
BACKUP_EXTRA_SITES = [
    x.strip().lower()
    for x in os.environ.get("SEARCH_BACKUP_EXTRA_SITES", "").split(",")
    if x.strip()
]

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 4
MAX_BACKOFF_SECONDS = 60

# См. "Шестое уточнение" в докстринге модуля: raw CLIP cosine similarity для реально
# релевантных пар текст-картинка типично лежит в диапазоне ~0.2-0.35, а не 0.5-0.85 -
# пороги пересчитаны под это и вынесены в окружение, чтобы их можно было донастроить
# без правки кода (например по перцентилю на собственной выборке сегментов).
SIM_ACCEPT_THRESHOLD = float(os.environ.get("SEARCH_SIM_ACCEPT_THRESHOLD", 0.30))
SIM_MIN_THRESHOLD = float(os.environ.get("SEARCH_SIM_MIN_THRESHOLD", 0.21))

# Относительная оценка (см. раздел "Относительная оценка CLIP" в докстринге модуля).
_REL_TOP_N_ENV = os.environ.get("SEARCH_REL_TOP_N")
REL_TOP_N_EXPLICIT = bool(_REL_TOP_N_ENV)
REL_TOP_N = int(_REL_TOP_N_ENV) if _REL_TOP_N_ENV else 10  # без явной установки - потолок
REL_TOP_N_FRACTION = 0.05
REL_MARGIN = float(os.environ.get("SEARCH_REL_MARGIN", 0.03))
FLAT_SPREAD = float(os.environ.get("SEARCH_FLAT_SPREAD", 0.03))
SCENE_ENCODE_BATCH = 64

CANDIDATES_PER_SITE = int(os.environ.get("SEARCH_CANDIDATES_PER_SITE", 5))

CLIP_MODEL_NAME = os.environ.get("SEARCH_CLIP_MODEL", "ViT-B-32-quickgelu")
CLIP_PRETRAINED = os.environ.get("SEARCH_CLIP_PRETRAINED", "openai")
CLIP_CONCURRENCY = int(os.environ.get("SEARCH_CLIP_CONCURRENCY", 2))

SEMAPHORE_DEFAULTS = {
    "pexels": int(os.environ.get("SEARCH_SEM_PEXELS", 5)),
    "pixabay": int(os.environ.get("SEARCH_SEM_PIXABAY", 5)),
    "wikimedia": int(os.environ.get("SEARCH_SEM_WIKIMEDIA", 10)),
    "nasa": int(os.environ.get("SEARCH_SEM_NASA", 10)),
    "loc": int(os.environ.get("SEARCH_SEM_LOC", 10)),
}
GLOBAL_SEGMENT_CONCURRENCY = int(os.environ.get("SEARCH_GLOBAL_CONCURRENCY", 40))

# Отдельный от семафора механизм - см. докстринг модуля, раздел "Третье"/"Четвёртое"
# допущение. Официальный лимит LOC - 20 запросов/мин у JSON/YAML API, час блокировки
# при превышении (working-within-limits). Умолч. 3.2с (~18.75/мин, ~6% запас от потолка,
# намеренно округлено В БОЛЬШУЮ сторону от математических 3.158с=5% запаса - см. подробное
# обоснование выше в SEARCH_LOC_MIN_INTERVAL_SECONDS).
LOC_MIN_INTERVAL_SECONDS = float(os.environ.get("SEARCH_LOC_MIN_INTERVAL_SECONDS", 3.2))

# ---------------------------------------------------------------------------
# Раунд 8 - отдельный лимитер для LOC-превью (fetch_preview_bytes), подтверждено
# официальной документацией, та же методика расчёта, что и в download.py.
#
# LOC_MIN_INTERVAL_SECONDS выше применяется ТОЛЬКО к search_loc (запрос к
# www.loc.gov/search/ - JSON/YAML API, лимит 20/мин). Превью-картинки кандидатов для
# CLIP-скоринга (fetch_preview_bytes, cand.preview_url) физически отдаются с
# tile.loc.gov/storage-services/... - той же самой категории "Media content", для
# которой официальная документация (https://www.loc.gov/apis/json-and-yaml/working-within-limits/)
# даёт отдельный, гораздо менее строгий лимит 150 запросов/мин (в 7.5 раза больше).
#
# ВАЖНО: до этой правки fetch_preview_bytes НЕ имела вообще НИКАКОГО интервального
# ограничения для LOC (только общий SEARCH_SEM_LOC=10 конкурентности через
# site_semaphores, да и то он реально применяется лишь внутри http_get_json - у самой
# fetch_preview_bytes семафора нет вовсе, ни общего, ни LOC-специфичного). Проверка
# на глаз (без реального прогона - сеть в этом окружении недоступна) показывает, что
# полагаться на "и так маловероятно" здесь нельзя: score_candidates обрабатывает
# кандидатов одного сегмента строго последовательно (простой for-await, без gather),
# но РАЗНЫЕ сегменты идут параллельно до GLOBAL_SEGMENT_CONCURRENCY=40 штук
# одновременно - т.е. реальная пиковая конкурентность LOC-превью зависит от того,
# сколько из этих 40 параллельных сегментов одновременно попали на LOC-кандидата, а
# не от SEARCH_SEM_LOC=10 (та цифра к превью не относится вообще). Это не подтверждено
# логами прогона (их нет), поэтому решение - не гадать, а добавить лимитер той же
# методикой, что и в LOC_FILE_MIN_INTERVAL_SECONDS из download.py: дешевле стоит
# лишних 0.45с на превью, чем рисковать часовым баном IP по недоказанному допущению.
#
# Конкурентность SEARCH_SEM_LOC=10 (используется в http_get_json для search_loc) не
# трогаем - она ограничивает другой запрос (сам поиск, не превью) и решает свою,
# независимую задачу, как и для wikimedia в download.py (интервал + конкурентность
# — два независимых ограничения, оба могут быть нужны одновременно).
LOC_PREVIEW_MIN_INTERVAL_SECONDS = float(
    os.environ.get("SEARCH_LOC_PREVIEW_MIN_INTERVAL_SECONDS", 0.45)
)
# 0.45с (~133 запроса/мин, ~11% запас от официального потолка 150/мин).

# ---------------------------------------------------------------------------
# Раунд 9 (Этап 1) - лимитер и семафор для превью Wikimedia Commons.
#
# По логу реального прогона 43 кандидата Wikimedia Commons были потеряны из-за 429
# на этапе скачивания превью картинок (upload.wikimedia.org). При глобальной
# параллельности 40 сегментов запросы картинок летели пачками без каких-либо
# ограничений (у Wikimedia в fetch_preview_bytes не было ни семафора, ни лимитера).
# Добавлены: интервал 0.35с (~170 запросов/мин) и семафор конкурентности на 3 слота.
# ---------------------------------------------------------------------------
WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS = float(
    os.environ.get("SEARCH_WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS", 0.35)
)
WIKIMEDIA_PREVIEW_CONCURRENCY = int(
    os.environ.get("SEARCH_WIKIMEDIA_PREVIEW_CONCURRENCY", 3)
)

# Таймаут поиска LOC: 30с предотвращает таймауты на холодных запросах (было 15с)
SEARCH_LOC_TIMEOUT_SECONDS = float(os.environ.get("SEARCH_LOC_TIMEOUT_SECONDS", 30))
SEARCH_LOC_MAX_RETRIES = max(1, int(os.environ.get("SEARCH_LOC_MAX_RETRIES", 3)))

PREVIEW_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

PREVIEW_REFERERS = {
    "pexels": "https://www.pexels.com/",
    "pixabay": "https://pixabay.com/",
    "wikimedia": "https://commons.wikimedia.org/",
    "nasa": "https://images.nasa.gov/",
    "loc": "https://www.loc.gov/",
}

SESSION_USER_AGENT = (
    "MediaSearchPipeline/1.0 "
    "(https://github.com/SOTONATORE/Download; contact: fordlababit@gmail.com)"
)

# Белый список LicenseShortName для Wikimedia Commons (регистронезависимо, по префиксу).
_WM_PD_RE = re.compile(r"^pd([-\s]|$)", re.IGNORECASE)
_WM_FREE_PREFIXES = ("cc0", "cc-zero", "cc by", "public domain", "fal", "gfdl")


def wikimedia_license_ok(license_short_name: str) -> bool:
    if not license_short_name:
        return False
    v = license_short_name.strip().lower()
    if _WM_PD_RE.match(v):
        return True
    return any(v.startswith(p) for p in _WM_FREE_PREFIXES)


def nasa_license_ok(item_data: dict) -> bool:
    return not item_data.get("copyright")


def loc_license_ok(item: dict) -> bool:
    """Эвристика: LOC не даёт единого чистого поля "свободно/не свободно", поэтому
    консервативно исключаем всё, что явно помечено access_restricted или содержит
    rights_advisory без явной фразы про отсутствие ограничений/public domain."""
    if item.get("access_restricted") is True:
        return False
    advisory = item.get("rights_advisory")
    if isinstance(advisory, list):
        advisory_text = " ".join(str(a) for a in advisory).lower()
    else:
        advisory_text = str(advisory or "").lower()
    if advisory_text:
        if "no known restriction" in advisory_text or "public domain" in advisory_text:
            return True
        return False
    return True


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


# ---------------------------------------------------------------------------
# Текстовый фильтр по сущностям (см. "Восьмое/Девятое уточнение" в докстринге)
# ---------------------------------------------------------------------------

_FUZZY_WORD_RATIO_THRESHOLD = float(os.environ.get("SEARCH_FUZZY_KEYWORD_RATIO", 0.78))
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _normalize_for_match(s: str) -> str:
    """NFKD + снятие комбинирующих диакритических знаков, плюс lower()."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


_MIN_FUZZY_WORD_LEN = 3


def _fuzzy_word_match(keyword_norm: str, text_words: list[str]) -> bool:
    if len(keyword_norm) < _MIN_FUZZY_WORD_LEN:
        return False
    for w in text_words:
        if len(w) < _MIN_FUZZY_WORD_LEN:
            continue
        if keyword_norm in w or w in keyword_norm:
            return True
        if difflib.SequenceMatcher(None, keyword_norm, w).ratio() >= _FUZZY_WORD_RATIO_THRESHOLD:
            return True
    return False


def text_matches_keywords(text: str, keywords: list[str]) -> bool:
    """Точное совпадение фразы -> нормализация NFKD -> пословное приближенное
    совпадение через difflib для слов длиннее _MIN_FUZZY_WORD_LEN."""
    if not keywords:
        return True
    t_raw = (text or "").lower()
    if any(kw.lower() in t_raw for kw in keywords if kw):
        return True

    norm_text = _normalize_for_match(text)
    if not norm_text:
        return False
    text_words = _WORD_RE.findall(norm_text)

    for kw in keywords:
        if not kw:
            continue
        kw_words = _WORD_RE.findall(_normalize_for_match(kw))
        if not kw_words:
            continue
        content_words = [w for w in kw_words if len(w) >= _MIN_FUZZY_WORD_LEN]
        if not content_words:
            if _normalize_for_match(kw) in norm_text:
                return True
            continue
        if all(_fuzzy_word_match(w, text_words) for w in content_words):
            return True
    return False


class FatalConfigError(RuntimeError):
    """Структурная ошибка конфигурации - приводит к остановке с кодом 1."""


# ---------------------------------------------------------------------------
# Диагностика воронки (raw -> лицензия -> сущности -> CLIP) по каждому сайту
# ---------------------------------------------------------------------------

@dataclass
class SiteStats:
    segments_attempted: int = 0
    raw_total: int = 0
    license_ok_total: int = 0
    keyword_ok_total: int = 0
    sent_to_clip_total: int = 0
    preview_missing_total: int = 0
    clip_error_total: int = 0
    clip_scored_total: int = 0
    clip_passed_total: int = 0
    clip_accept_total: int = 0
    accepted_total: int = 0
    rejected_foreign_total: int = 0
    best_effort_total: int = 0
    score_sum: float = 0.0
    best_score: float = 0.0

    def record_score(self, sim: float) -> None:
        self.clip_scored_total += 1
        self.score_sum += sim
        if sim > self.best_score:
            self.best_score = sim
        if sim >= SIM_ACCEPT_THRESHOLD:
            self.clip_accept_total += 1
        if sim >= SIM_MIN_THRESHOLD:
            self.clip_passed_total += 1

    @property
    def avg_score(self) -> float:
        return (self.score_sum / self.clip_scored_total) if self.clip_scored_total else 0.0


def log_site_stats_summary(site_stats: dict) -> None:
    if not site_stats:
        return
    logging.info("=" * 100)
    logging.info("СВОДКА ПО ВОРОНКЕ ФИЛЬТРАЦИИ (диагностика, откуда берутся нули):")
    logging.info(
        "%-18s %6s %7s %8s %9s %8s %9s %8s %8s %8s %7s %7s %8s %7s %8s",
        "сайт", "сегм.", "raw", "лиценз.", "keyword", "->CLIP", "нет прев.",
        "scored", f">={SIM_MIN_THRESHOLD:.2f}", f">={SIM_ACCEPT_THRESHOLD:.2f}", "avg", "best",
        "принято", "чужие", "best-eff",
    )
    for key in sorted(site_stats.keys()):
        s = site_stats[key]
        logging.info(
            "%-18s %6d %7d %8d %9d %8d %9d %8d %8d %8d %7.3f %7.3f %8d %7d %8d",
            key, s.segments_attempted, s.raw_total, s.license_ok_total,
            s.keyword_ok_total, s.sent_to_clip_total, s.preview_missing_total,
            s.clip_scored_total, s.clip_passed_total, s.clip_accept_total,
            s.avg_score, s.best_score,
            s.accepted_total, s.rejected_foreign_total, s.best_effort_total,
        )
    logging.info(
        "Как читать: raw=0 -> сайт вообще ничего не вернул по запросу (сеть/сам API/лимит). "
        "лиценз.=0 при raw>0 -> все кандидаты отсеяны лицензионным фильтром. "
        "keyword=0 при лиценз.>0 -> entity_keywords/is_entity слишком узкие или не совпадают "
        "с текстом кандидатов. нет_прев.=raw (или близко) -> превью не скачиваются (сайт "
        "блокирует PREVIEW_HEADERS/Referer/хотлинкинг - точный статус-код и тело ответа по "
        "каждому провалу теперь всегда пишется отдельным логгером 'search.fetch_preview_bytes' "
        "на уровне DEBUG, см. его вывод выше). scored>0, но "
        "принято = кандидаты, принятые по рангу своей сцены среди всех сегментов или по "
        "абсолютному порогу; чужие = отсеяны, т.к. другая сцена подходит им заметно лучше "
        "своей; best-eff = никто не принят, взят лучший по own_sim (полный автомат). "
        "avg/best - это сходство со СВОЕЙ сценой; низкие значения -> CLIP отрабатывает, но ничего не "
        "совпадает по смыслу - либо сам CLIP настроен неверно (см. 'Пятое уточнение' в "
        "докстринге модуля про QuickGELU), либо запросы от generate_queries.py слишком "
        "специфичны/не по делу. Учтите: raw CLIP similarity для здоровых совпадений обычно "
        "лежит в диапазоне ~0.2-0.35 (см. 'Шестое уточнение') - это НЕ то же самое, что "
        "similarity софтмакса/температурного скейлинга."
    )
    logging.info("=" * 100)


# ---------------------------------------------------------------------------
# Глобальный rate-limiter (минимальный интервал между запросами)
# ---------------------------------------------------------------------------

RATE_LIMITER_DEBUG_LOGGER = logging.getLogger("search.rate_limiter")
RATE_LIMITER_DEBUG_LOGGER.setLevel(logging.DEBUG)
CLIP_TIMING_DEBUG_LOGGER = logging.getLogger("search.clip_timing")
CLIP_TIMING_DEBUG_LOGGER.setLevel(logging.DEBUG)


class RateLimiter:
    """Гарантирует минимальный интервал между НАЧАЛОМ двух последовательных запросов,
    глобально на весь запуск - в отличие от asyncio.Semaphore, который ограничивает
    только число одновременно летящих запросов."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval = max(0.0, min_interval_seconds)
        self._lock = asyncio.Lock()
        self._last_start_ts: Optional[float] = None

    async def wait_turn(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._last_start_ts is not None:
                elapsed = now - self._last_start_ts
                remaining = self.min_interval - elapsed
                if remaining > 0:
                    wait_start = time.monotonic()
                    await asyncio.sleep(remaining)
                    actual_wait = time.monotonic() - wait_start
                    RATE_LIMITER_DEBUG_LOGGER.debug(
                        "RateLimiter.wait_turn: реально ждал %.3fs (запрошено %.3fs, "
                        "min_interval=%.2fs)", actual_wait, remaining, self.min_interval,
                    )
                    now = loop.time()
            self._last_start_ts = now


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------

@dataclass
class SegmentSpec:
    index: int
    scene: str
    sites: list[str]
    query_narrow: str
    query_medium: str
    query_broad: Optional[str]
    type: str  # "image" | "video"
    is_entity: bool
    entity_keywords: list[str]


@dataclass
class Candidate:
    site: str
    cand_id: str
    text: str
    license_ok: bool
    preview_url: Optional[str]
    page_url: Optional[str]
    final_url_resolver: Optional[Callable[["Context"], Awaitable[Optional[str]]]] = None
    similarity: float = 0.0  # = own_sim (для сортировки)
    own_sim: float = 0.0
    rank: int = 0
    accepted: bool = False
    reject_reason: str = ""
    stats_key: Optional[str] = None
    variant: Optional[str] = None  # narrow | medium | broad


@dataclass
class Context:
    session: "aiohttp.ClientSession"
    pexels_api_key: str
    pixabay_api_key: str
    site_semaphores: dict
    global_semaphore: asyncio.Semaphore
    clip_semaphore: asyncio.Semaphore
    used_files_lock: asyncio.Lock
    used_files: set = field(default_factory=set)
    exhausted_sites: set = field(default_factory=set)
    search_cache: dict = field(default_factory=dict)
    search_cache_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    clip: "ClipScorer" = None
    rate_limiters: dict = field(default_factory=dict)
    site_stats: dict = field(default_factory=dict)
    format_rejects: FormatRejectStats = field(default_factory=FormatRejectStats)
    primary_cands: dict = field(default_factory=dict)
    rel_top_n: int = 3
    choice_reasons: Counter = field(default_factory=Counter)
    choice_own_sims: list = field(default_factory=list)
    backup_reasons: Counter = field(default_factory=Counter)
    backup_own_sims: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

@dataclass
class RankResult:
    accepted: bool
    own_sim: float
    rank: int
    reason: str  # rank | abs | floor | foreign | flat | rank_miss


def effective_top_n(n_segments: int, top_n: int = REL_TOP_N, explicit: bool = REL_TOP_N_EXPLICIT) -> int:
    """Явный SEARCH_REL_TOP_N берётся как есть; иначе max(3, min(top_n, ceil(5% сегментов)))."""
    if explicit:
        return max(1, top_n)
    return max(3, min(top_n, math.ceil(REL_TOP_N_FRACTION * max(1, n_segments))))


def rank_decision(
    sims_all: Sequence[float], own_row: int, top_n: int,
    margin: float = REL_MARGIN, min_abs: float = SIM_MIN_THRESHOLD,
    flat_spread: float = FLAT_SPREAD, accept_abs: float = SIM_ACCEPT_THRESHOLD,
) -> RankResult:
    """Ядро относительной оценки, без сети и ctx. sims_all - сходства кандидата со scene
    всех сегментов, own_row - строка его собственного сегмента."""
    own = float(sims_all[own_row])
    rank = 1 + sum(1 for i, x in enumerate(sims_all) if i != own_row and x > own)
    if own < min_abs:
        return RankResult(False, own, rank, "floor")
    if len(sims_all) > 1 and (max(sims_all) - min(sims_all)) < flat_spread:
        # однотемный ролик: относительный критерий неинформативен, решает только абсолютный
        ok = own >= accept_abs
        return RankResult(ok, own, rank, "abs" if ok else "flat")
    if own >= accept_abs:
        return RankResult(True, own, rank, "abs")
    max_other = max((x for i, x in enumerate(sims_all) if i != own_row), default=None)
    if max_other is not None and own + margin < max_other:
        return RankResult(False, own, rank, "foreign")
    if rank <= top_n:
        return RankResult(True, own, rank, "rank")
    return RankResult(False, own, rank, "rank_miss")


def rank_candidate(
    sims_all: Sequence[float], own_row: int, top_n: int,
    margin: float = REL_MARGIN, min_abs: float = SIM_MIN_THRESHOLD,
    flat_spread: float = FLAT_SPREAD, accept_abs: float = SIM_ACCEPT_THRESHOLD,
) -> tuple[bool, float, int]:
    r = rank_decision(sims_all, own_row, top_n, margin, min_abs, flat_spread, accept_abs)
    return r.accepted, r.own_sim, r.rank


def compute_sims(scene_matrix: Any, image_vec: Any) -> list[float]:
    """Косинусные сходства (векторы нормализованы) картинки со всеми scene."""
    return (scene_matrix @ image_vec).tolist()


class EmbeddingCache:
    """Кэш нормализованных векторов превью по ключу (site, cand_id). Один и тот же ключ
    кодируется один раз даже при параллельных запросах (общий future на ключ).
    Хранит только вектор (512 float32 ~ 2 КБ), не картинку и не байты."""

    def __init__(self) -> None:
        self._store: dict = {}
        self._pending: dict = {}
        self.hits = 0
        self.computed = 0

    def __len__(self) -> int:
        return len(self._store)

    async def get_or_compute(self, key: tuple, compute: Callable[[], Awaitable[Any]]) -> Any:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        fut = self._pending.get(key)
        if fut is not None:
            self.hits += 1
            return await fut
        fut = asyncio.get_running_loop().create_future()
        self._pending[key] = fut
        try:
            val = await compute()
            if val is not None:
                self._store[key] = val
                self.computed += 1
            fut.set_result(val)
            return val
        except BaseException:
            if not fut.done():
                fut.set_result(None)
            raise
        finally:
            self._pending.pop(key, None)


class _ClipEncodeError(Exception):
    pass


class ClipScorer:
    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load_lock = asyncio.Lock()
        self.scene_matrix = None  # torch float32 [n_segments, dim], строки нормализованы
        self.scene_rows: dict[int, int] = {}
        self.cache = EmbeddingCache()

    async def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            logging.info(
                "Загружаю CLIP-модель %s (%s) - при первом запуске без кэша качается "
                "с интернета, дальше держите её в actions/cache.",
                self.model_name, self.pretrained,
            )
            loop = asyncio.get_running_loop()
            model, _, preprocess = await loop.run_in_executor(
                None,
                functools.partial(
                    open_clip.create_model_and_transforms,
                    self.model_name, pretrained=self.pretrained,
                ),
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model, self._preprocess, self._tokenizer = model, preprocess, tokenizer
            logging.info("CLIP-модель загружена.")

    def _encode_texts_sync(self, texts: list[str]):
        tokens = self._tokenizer([t[:300] for t in texts])
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float()

    async def encode_scenes(self, scenes: list[tuple[int, str]]) -> None:
        """Один раз кодирует scene всех сегментов батчами и строит матрицу + индекс строк."""
        await self.ensure_loaded()
        loop = asyncio.get_running_loop()
        parts = []
        for i in range(0, len(scenes), SCENE_ENCODE_BATCH):
            batch = [t for _, t in scenes[i:i + SCENE_ENCODE_BATCH]]
            parts.append(await loop.run_in_executor(None, self._encode_texts_sync, batch))
        self.scene_matrix = torch.cat(parts, dim=0).float().contiguous()
        self.scene_rows = {idx: row for row, (idx, _) in enumerate(scenes)}
        logging.info("CLIP: закодировано scene: %s (батчи по %s).", len(scenes), SCENE_ENCODE_BATCH)

    def encode_image_sync(self, image_bytes: bytes):
        """Только кодирует и нормализует картинку (текст здесь не кодируется)."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            feats = self._model.encode_image(image_input)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).float().cpu()

    async def encode_image(self, image_bytes: bytes):
        await self.ensure_loaded()
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        try:
            return await loop.run_in_executor(None, self.encode_image_sync, image_bytes)
        finally:
            elapsed = time.monotonic() - start
            CLIP_TIMING_DEBUG_LOGGER.debug("CLIP-кодирование картинки (%s, %s): %.3fs", self.model_name, self.pretrained, elapsed)

    def similarities(self, image_vec) -> list[float]:
        return compute_sims(self.scene_matrix, image_vec)


# ---------------------------------------------------------------------------
# HTTP-хелпер с ретраями/бэкоффом + учёт 429-исчерпания
# ---------------------------------------------------------------------------

async def http_get_json(
    ctx: Context,
    site: str,
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    treat_429_as_exhaustion: bool = False,
) -> Any:
    if site in ctx.exhausted_sites:
        return None

    rate_limiter = ctx.rate_limiters.get(site)

    def _request_id() -> str:
        for k in ("q", "query", "srsearch"):
            if params and params.get(k) is not None:
                return f"{url} {k}={str(params[k])[:80]}"
        return url

    def _redact(text: str) -> str:
        for k in ("key", "api_key", "apikey", "token", "access_token"):
            v = (params or {}).get(k)
            if v:
                text = text.replace(str(v), "***")
        for k, v in (headers or {}).items():
            if k.lower() in ("authorization", "x-api-key", "api-key") and v:
                text = text.replace(str(v), "***")
        return re.sub(r"(?i)\b(key|api_key|apikey|token|access_token)=[^&\s'\"]+", r"\1=***", text)

    # Таймаут поиска LOC: 30с по умолчанию (раньше 15с приводило к постоянным TimeoutError)
    if site == "loc":
        max_attempts = SEARCH_LOC_MAX_RETRIES
        timeout_seconds = SEARCH_LOC_TIMEOUT_SECONDS
    else:
        max_attempts = MAX_RETRIES
        timeout_seconds = 30

    request_id = _request_id()
    last_error: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        has_next = attempt < max_attempts
        backoff = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
        retry_after: Optional[float] = None

        async with ctx.site_semaphores[site]:
            if rate_limiter is not None:
                await rate_limiter.wait_turn()
            attempt_started = time.monotonic()
            try:
                async with ctx.session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                ) as resp:
                    status = resp.status

                    if status == 429:
                        if treat_429_as_exhaustion:
                            logging.warning(
                                "%s: получен 429 - помечаю сайт исчерпанным до конца "
                                "текущего запуска.", site,
                            )
                            ctx.exhausted_sites.add(site)
                            return None
                        logging.warning(
                            "%s: 429, попытка %s/%s, %s.", site, attempt, max_attempts,
                            f"жду {backoff:.1f}s" if has_next else "попыток больше нет",
                        )
                        last_error = RuntimeError("429 Too Many Requests")
                        retry_after = backoff

                    elif status in (401, 403):
                        text = await resp.text()
                        if site in ("pexels", "pixabay"):
                            raise FatalConfigError(
                                f"{site}: HTTP {status} - похоже на невалидный API-ключ. "
                                f"Тело ответа: {text[:300]}"
                            )
                        logging.error(
                            "%s: HTTP %s - не ретраю (не временная ошибка). Тело: %s",
                            site, status, text[:300],
                        )
                        return None

                    elif status >= 500:
                        logging.warning(
                            "%s: HTTP %s (попытка %s/%s), %s.", site, status, attempt, max_attempts,
                            f"жду {backoff:.1f}s" if has_next else "попыток больше нет",
                        )
                        last_error = RuntimeError(f"HTTP {status}")
                        retry_after = backoff

                    elif status != 200:
                        text = await resp.text()
                        logging.warning(
                            "%s: неожиданный статус %s (попытка %s/%s): %s",
                            site, status, attempt, max_attempts, text[:200],
                        )
                        last_error = RuntimeError(f"HTTP {status}: {text[:300]}")
                        retry_after = backoff

                    else:
                        try:
                            data = await resp.json(content_type=None)
                        except aiohttp.ContentTypeError:
                            text_preview = (await resp.text())[:200]
                            if treat_429_as_exhaustion:
                                logging.warning(
                                    "%s: получен не-JSON ответ (похоже на CAPTCHA/rate-limit "
                                    "страницу вместо API-ответа) - помечаю сайт исчерпанным до "
                                    "конца текущего запуска. Превью тела: %s",
                                    site, text_preview,
                                )
                                ctx.exhausted_sites.add(site)
                                return None
                            logging.warning(
                                "%s: не-JSON ответ при статусе 200 (попытка %s/%s), похоже на "
                                "CAPTCHA/перегрузку. Превью: %s", site, attempt, max_attempts, text_preview,
                            )
                            last_error = RuntimeError(f"non-JSON 200 response: {text_preview}")
                            retry_after = backoff
                        else:
                            if attempt > 1:
                                logging.info(
                                    "%s: успех с попытки %s/%s (%s)",
                                    site, attempt, max_attempts, request_id,
                                )
                            return data

            except FatalConfigError:
                raise
            except (aiohttp.ClientError, aiohttp.ClientPayloadError,
                    aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as e:
                last_error = e
                elapsed = time.monotonic() - attempt_started
                logging.warning(
                    "%s: сетевая ошибка (попытка %s/%s) [%s: %s] запрос=%s, длительность попытки %.1fs. %s",
                    site, attempt, max_attempts, type(e).__name__,
                    _redact(repr(e)), request_id, elapsed,
                    f"Жду {backoff:.1f}s." if has_next else "Попыток больше нет.",
                )
                retry_after = backoff

        if retry_after is not None and has_next:
            await asyncio.sleep(retry_after + random.uniform(0, 1))

    logging.error(
        "%s: запрос не удался после %s попыток (%s): %s",
        site, max_attempts, request_id,
        _redact(f"{type(last_error).__name__}: {last_error!r}"),
    )
    return None


async def cached_search(
    ctx: Context, site: str, media_type: str, query: str,
    fetch_coro_factory: Callable[[], Awaitable[list]],
) -> list:
    """Single-flight кэш по (site, media_type, query): если запрос уже в процессе -
    ждём его же, а не дублируем."""
    key = (site, media_type, query)
    async with ctx.search_cache_lock:
        task = ctx.search_cache.get(key)
        if task is None:
            task = asyncio.ensure_future(fetch_coro_factory())
            ctx.search_cache[key] = task
    return await task


# ---------------------------------------------------------------------------
# Поиск по сайтам
# ---------------------------------------------------------------------------

async def search_pexels(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        if "pexels" in ctx.exhausted_sites:
            return []
        url = "https://api.pexels.com/videos/search" if media_type == "video" else "https://api.pexels.com/v1/search"
        headers = {"Authorization": ctx.pexels_api_key}
        params = {"query": query, "per_page": 15}
        data = await http_get_json(ctx, "pexels", url, headers=headers, params=params, treat_429_as_exhaustion=True)
        result: list[Candidate] = []
        if not data:
            return result
        if media_type == "video":
            for v in data.get("videos", []):
                preview = v.get("image")
                if not preview:
                    pics = v.get("video_pictures") or []
                    preview = pics[0]["picture"] if pics else None
                result.append(Candidate(
                    site="pexels", cand_id=str(v["id"]), text="",
                    license_ok=True, preview_url=preview, page_url=v.get("url"),
                ))
        else:
            for p in data.get("photos", []):
                src = p.get("src") or {}
                preview = src.get("medium") or src.get("small") or src.get("original")
                result.append(Candidate(
                    site="pexels", cand_id=str(p["id"]), text=p.get("alt") or "",
                    license_ok=True, preview_url=preview, page_url=p.get("url"),
                ))
        return result

    return await cached_search(ctx, "pexels", media_type, query, _do)


async def search_pixabay(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        if "pixabay" in ctx.exhausted_sites:
            return []
        url = "https://pixabay.com/api/videos/" if media_type == "video" else "https://pixabay.com/api/"
        params = {"key": ctx.pixabay_api_key, "q": query, "per_page": 20}
        data = await http_get_json(ctx, "pixabay", url, params=params, treat_429_as_exhaustion=True)
        result: list[Candidate] = []
        if not data:
            return result
        for hit in data.get("hits", []):
            page_url = hit.get("pageURL")
            tags = hit.get("tags", "")
            if media_type == "video":
                videos = hit.get("videos") or {}
                preview = None
                for size in ("tiny", "small", "medium", "large"):
                    preview = (videos.get(size) or {}).get("thumbnail")
                    if preview:
                        break
            else:
                preview = hit.get("previewURL") or hit.get("webformatURL")
            result.append(Candidate(
                site="pixabay", cand_id=str(hit["id"]), text=tags,
                license_ok=True, preview_url=preview, page_url=page_url,
            ))
        return result

    return await cached_search(ctx, "pixabay", media_type, query, _do)


COMMONS_VIDEO_EXTENSIONS = frozenset({"webm", "ogv", "mpg", "mpeg"})
_commons_video_skip_logged = False


async def search_wikimedia(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    global _commons_video_skip_logged
    if media_type == "video" and not (COMMONS_VIDEO_EXTENSIONS & VIDEO_EXTENSIONS):
        if not _commons_video_skip_logged:
            _commons_video_skip_logged = True
            logging.info(
                "wikimedia: видео пропускается целиком - форматы Commons (%s) не входят в "
                "белый список видео (%s)",
                ", ".join(sorted(COMMONS_VIDEO_EXTENSIONS)), ", ".join(sorted(VIDEO_EXTENSIONS)),
            )
        return []

    async def _do() -> list[Candidate]:
        if media_type == "video":
            search_query = f"{query} filetype:video"
        else:
            # filetype:bitmap исключает сканы документов (PDF, DjVu), аудио и векторные SVG,
            # экономя квоту выдачи srlimit=20 под реальные изображения
            search_query = f"{query} filetype:bitmap"

        params = {
            "action": "query", "list": "search", "srsearch": search_query,
            "srnamespace": 6, "srlimit": 20, "format": "json",
        }
        data = await http_get_json(ctx, "wikimedia", "https://commons.wikimedia.org/w/api.php", params=params)
        if not data:
            return []
        hits = data.get("query", {}).get("search", [])

        kind = "video" if media_type == "video" else "photo"
        kept = []
        for h in hits:
            ext = ext_from_name(h.get("title", ""))
            if is_allowed_ext(ext, kind):
                kept.append(h)
            else:
                ctx.format_rejects.add("wikimedia", ext)
        hits = kept
        if not hits:
            return []

        titles = [h["title"] for h in hits[:20]]
        snippet_by_title = {h["title"]: strip_html(h.get("snippet", "")) for h in hits}
        iiparams = {
            "action": "query", "prop": "imageinfo",
            "titles": "|".join(titles),
            "iiprop": "url|extmetadata",
            "iiurlwidth": 800,
            "format": "json",
        }
        info_data = await http_get_json(ctx, "wikimedia", "https://commons.wikimedia.org/w/api.php", params=iiparams)
        result: list[Candidate] = []
        if not info_data:
            return result
        pages = info_data.get("query", {}).get("pages", {}) or {}
        for _, page in pages.items():
            title = page.get("title")
            infos = page.get("imageinfo") or []
            if not title or not infos:
                continue
            info = infos[0]
            direct_url = info.get("url")
            thumb_url = info.get("thumburl")
            if not direct_url:
                continue
            if media_type == "video" and not thumb_url:
                continue
            extm = info.get("extmetadata", {}) or {}
            license_short = (extm.get("LicenseShortName") or {}).get("value", "")
            description = strip_html((extm.get("ImageDescription") or {}).get("value", ""))
            text = " ".join(filter(None, [title, snippet_by_title.get(title, ""), description]))
            wiki_page_url = "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")
            result.append(Candidate(
                site="wikimedia", cand_id=title, text=text,
                license_ok=wikimedia_license_ok(license_short),
                preview_url=thumb_url or direct_url, page_url=wiki_page_url,
            ))
        return result

    return await cached_search(ctx, "wikimedia", media_type, query, _do)


def _nasa_manifest_resolver(manifest_href: Optional[str], media_type: str):
    if not manifest_href:
        return None

    async def _resolve(ctx: Context) -> Optional[str]:
        files = await http_get_json(ctx, "nasa", manifest_href)
        if not files or not isinstance(files, list):
            return None
        kind = "video" if media_type == "video" else "photo"
        files = [f for f in files if isinstance(f, str)]
        candidates = [f for f in files if is_allowed_ext(ext_from_name(f), kind)]

        def rank(f: str) -> int:
            fl = f.lower()
            if "~orig" in fl:
                return 3
            if "~large" in fl:
                return 2
            if "~medium" in fl:
                return 1
            return 0

        if not candidates:
            with_ext = [f for f in files if ext_from_name(f)]
            if with_ext:
                best = max(with_ext, key=rank)
                ctx.format_rejects.add("nasa", ext_from_name(best))
            return None

        candidates.sort(key=rank, reverse=True)
        return candidates[0]

    return _resolve


async def search_nasa(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        params = {"q": query, "media_type": media_type}
        data = await http_get_json(ctx, "nasa", "https://images-api.nasa.gov/search", params=params)
        result: list[Candidate] = []
        if not data:
            return result
        items = (data.get("collection") or {}).get("items", []) or []
        for item in items:
            datas = item.get("data") or []
            if not datas:
                continue
            d0 = datas[0]
            links = item.get("links") or []
            preview = None
            for l in links:
                if l.get("rel") == "preview":
                    preview = l.get("href")
                    break
            if not preview and links:
                preview = links[0].get("href")
            text = " ".join(filter(None, [
                d0.get("title", ""), d0.get("description", ""),
                " ".join(d0.get("keywords") or []),
            ]))
            manifest_href = item.get("href")
            nasa_id = d0.get("nasa_id") or manifest_href
            if not nasa_id:
                continue
            result.append(Candidate(
                site="nasa", cand_id=nasa_id, text=text,
                license_ok=nasa_license_ok(d0), preview_url=preview, page_url=None,
                final_url_resolver=_nasa_manifest_resolver(manifest_href, media_type),
            ))
        return result

    return await cached_search(ctx, "nasa", media_type, query, _do)


async def search_loc(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        params: dict = {"q": query, "fo": "json", "c": 20}
        if media_type == "video":
            params["fa"] = "partof:online video"
        data = await http_get_json(
            ctx, "loc", "https://www.loc.gov/search/", params=params,
            treat_429_as_exhaustion=True,
        )
        result: list[Candidate] = []
        if not data:
            return result
        for item in data.get("results", []) or []:
            item_id = item.get("id")
            if not item_id:
                continue

            # Фильтрация не-айтемов: отсекаем виртуальные выставки (/exhibits/), блоги (/blogs/)
            # и служебные страницы, оставляя только оцифрованные каталожные объекты (/item/, /resource/).
            if not ("/item/" in item_id or "/resource/" in item_id):
                continue

            title = item.get("title", "")
            desc = item.get("description")
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc)

            # Обогащение текста для фильтра сущностей рубриками subject
            subjects = item.get("subject") or []
            if isinstance(subjects, list):
                subj_text = " ".join(str(s) for s in subjects)
            else:
                subj_text = str(subjects or "")

            text = " ".join(filter(None, [title, desc or "", subj_text]))
            images = item.get("image_url") or []

            if not any(isinstance(u, str) and is_allowed_ext(ext_from_name(u), "photo") for u in images):
                first = next((u for u in images if isinstance(u, str)), "")
                ctx.format_rejects.add("loc", ext_from_name(first))
                continue
            preview = next(
                u for u in images if isinstance(u, str) and is_allowed_ext(ext_from_name(u), "photo")
            )
            result.append(Candidate(
                site="loc", cand_id=item_id, text=text,
                license_ok=loc_license_ok(item), preview_url=preview, page_url=item_id,
            ))
        return result

    return await cached_search(ctx, "loc", media_type, query, _do)


SITE_SEARCH_FUNCS: dict = {
    "pexels": search_pexels,
    "pixabay": search_pixabay,
    "wikimedia": search_wikimedia,
    "nasa": search_nasa,
    "loc": search_loc,
}


# ---------------------------------------------------------------------------
# Превью + CLIP-скоринг + финализация URL
# ---------------------------------------------------------------------------

PREVIEW_DEBUG_LOGGER = logging.getLogger("search.fetch_preview_bytes")
PREVIEW_DEBUG_LOGGER.setLevel(logging.DEBUG)


async def fetch_preview_bytes(ctx: Context, cand: Candidate) -> Optional[bytes]:
    if not cand.preview_url:
        PREVIEW_DEBUG_LOGGER.debug(
            "Превью %s/%s: у кандидата нет preview_url вообще (текст кандидата: %r) - "
            "запрос в сеть не уходит, поиск на своей стороне не вернул URL превью.",
            cand.site, cand.cand_id, (cand.text or "")[:120],
        )
        return None

    headers = dict(PREVIEW_HEADERS)
    if cand.site == "wikimedia":
        headers["User-Agent"] = SESSION_USER_AGENT
    referer = cand.page_url if (cand.site == "wikimedia" and cand.page_url) else PREVIEW_REFERERS.get(cand.site)
    if referer:
        headers["Referer"] = referer
        parsed = urlparse(referer)
        if parsed.scheme and parsed.netloc:
            headers["Origin"] = f"{parsed.scheme}://{parsed.netloc}"

    last_status: Optional[int] = None
    last_body_preview = ""
    loc_preview_limiter = ctx.rate_limiters.get("loc_preview") if cand.site == "loc" else None
    wm_preview_limiter = ctx.rate_limiters.get("wikimedia_preview") if cand.site == "wikimedia" else None
    wm_preview_sem = ctx.site_semaphores.get("wikimedia_preview") if cand.site == "wikimedia" else None

    for attempt in range(1, 3):
        if loc_preview_limiter is not None:
            await loc_preview_limiter.wait_turn()
        if wm_preview_limiter is not None:
            await wm_preview_limiter.wait_turn()

        try:
            async def _do_req():
                async with ctx.session.get(
                    cand.preview_url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    raw_bytes = await resp.read()
                    return resp.status, resp.headers.get("Retry-After"), raw_bytes

            if wm_preview_sem is not None:
                async with wm_preview_sem:
                    status, retry_after, body = await _do_req()
            else:
                status, retry_after, body = await _do_req()

            if status == 200:
                return body

            last_status = status
            try:
                last_body_preview = body[:200].decode("utf-8", errors="replace").replace("\n", " ")
            except Exception:
                last_body_preview = "<не удалось декодировать тело>"

            if status == 429:
                try:
                    sleep_sec = float(retry_after) if retry_after else (3.0 * attempt + random.uniform(0.5, 1.5))
                except ValueError:
                    sleep_sec = 3.0 * attempt + random.uniform(0.5, 1.5)
                PREVIEW_DEBUG_LOGGER.debug(
                    "Превью %s/%s: HTTP 429 при GET %s (попытка %s/2, Retry-After=%s, жду %.1fs). Тело: %s",
                    cand.site, cand.cand_id, cand.preview_url, attempt, retry_after, sleep_sec, last_body_preview,
                )
                if attempt < 2:
                    await asyncio.sleep(sleep_sec)
                    continue
                return None

            elif status in (500, 502, 503, 504):
                PREVIEW_DEBUG_LOGGER.debug(
                    "Превью %s/%s: HTTP %s при GET %s (попытка %s/2). Тело: %s",
                    cand.site, cand.cand_id, status, cand.preview_url, attempt, last_body_preview,
                )
                if attempt < 2:
                    await asyncio.sleep(2.0 * attempt)
                    continue
                return None
            else:
                PREVIEW_DEBUG_LOGGER.debug(
                    "Превью %s/%s: HTTP %s при GET %s (попытка %s/2, Referer=%s). Тело: %s",
                    cand.site, cand.cand_id, status, cand.preview_url, attempt, referer, last_body_preview,
                )
                return None

        except (aiohttp.ClientError, aiohttp.ClientPayloadError,
                aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as e:
            PREVIEW_DEBUG_LOGGER.debug(
                "Превью %s/%s: сетевая ошибка (попытка %s/2) при GET %s: %s: %s",
                cand.site, cand.cand_id, attempt, cand.preview_url, type(e).__name__, e,
            )
            if attempt < 2:
                await asyncio.sleep(1.5 * attempt + random.uniform(0, 0.5))

    if last_status is not None:
        PREVIEW_DEBUG_LOGGER.debug(
            "Превью %s/%s: не удалось скачать после всех попыток, последний статус %s, тело: %s",
            cand.site, cand.cand_id, last_status, last_body_preview,
        )
    return None


async def score_candidates(
    ctx: Context, candidates: list[Candidate],
    seg_index: Optional[int] = None, stats_key: Optional[str] = None,
    variant_name: Optional[str] = None,
) -> list[Candidate]:
    """Оценивает ВСЕХ кандидатов с превью относительно scene всех сегментов. Никого не
    отбрасывает по порогу: решение принято/не принято записано в копию Candidate
    (accepted, own_sim, rank, reject_reason); similarity = own_sim. Возвращает по убыванию own_sim."""
    key = stats_key or (candidates[0].site if candidates else "unknown")
    stats = ctx.site_stats.setdefault(key, SiteStats())
    own_row = ctx.clip.scene_rows[seg_index]
    n_with_preview = 0

    scored: list[Candidate] = []
    for cand in candidates:
        async def compute(cand=cand):
            preview_bytes = await fetch_preview_bytes(ctx, cand)
            if preview_bytes is None:
                return None
            async with ctx.clip_semaphore:
                try:
                    return await ctx.clip.encode_image(preview_bytes)
                except Exception as e:
                    raise _ClipEncodeError(str(e)) from e

        try:
            vec = await ctx.clip.cache.get_or_compute((cand.site, cand.cand_id), compute)
        except _ClipEncodeError as e:
            logging.debug("CLIP не смог закодировать %s/%s: %s", cand.site, cand.cand_id, e)
            stats.clip_error_total += 1
            continue
        if vec is None:
            stats.preview_missing_total += 1
            continue
        stats.sent_to_clip_total += 1
        n_with_preview += 1
        res = rank_decision(
            ctx.clip.similarities(vec), own_row, ctx.rel_top_n,
            REL_MARGIN, SIM_MIN_THRESHOLD, FLAT_SPREAD, SIM_ACCEPT_THRESHOLD,
        )
        stats.record_score(res.own_sim)
        if res.accepted:
            stats.accepted_total += 1
        elif res.reason == "foreign":
            stats.rejected_foreign_total += 1
        # копия: Candidate может лежать в общем search_cache и оцениваться разными сегментами
        scored.append(dc_replace(
            cand, similarity=res.own_sim, own_sim=res.own_sim, rank=res.rank,
            accepted=res.accepted, reject_reason=res.reason, stats_key=key, variant=variant_name,
        ))
    scored.sort(key=lambda c: c.own_sim, reverse=True)

    if candidates and not scored:
        logging.info(
            "Сегмент %s/%s: %s кандидатов, но ни для одного не удалось получить вектор превью "
            "(скачивание/кодирование) - проверьте лог 'search.fetch_preview_bytes' на уровне DEBUG.",
            seg_index, key, len(candidates),
        )
    elif scored and not any(c.accepted for c in scored):
        logging.info(
            "Сегмент %s/%s: %s кандидатов оценено, принятых нет (лучший own_sim=%.3f, rank=%s, причина: %s).",
            seg_index, key, len(scored), scored[0].own_sim, scored[0].rank, scored[0].reject_reason,
        )
    return scored


async def finalize_candidate(ctx: Context, cand: Candidate) -> Optional[str]:
    if cand.final_url_resolver is not None:
        try:
            return await cand.final_url_resolver(ctx)
        except Exception as e:
            logging.warning("Не удалось финализировать %s/%s: %s", cand.site, cand.cand_id, e)
            return None
    return cand.page_url


async def try_claim_backup(ctx: Context, tail: list[Candidate], primary_site: str) -> Optional[str]:
    """Выбирает РОВНО ОДНОГО backup-кандидата.
    Приоритет: кандидат с сайта, ОТЛИЧНОГО от сайта primary.
    Для LOC брать backup с того же сайта строго запрещено: при часовом бане IP
    оба кандидата гарантированно погибнут. Возвращаем None, чтобы независимый
    backup нашел run_backup_pass."""
    backup_cand = next((c for c in tail if c.site != primary_site), None)
    if backup_cand is None and primary_site != "loc":
        backup_cand = next((c for c in tail if c.site == primary_site), None)
    if backup_cand is None:
        return None

    key = (backup_cand.site, backup_cand.cand_id)
    async with ctx.used_files_lock:
        if key in ctx.used_files:
            return None
        ctx.used_files.add(key)
    url = await finalize_candidate(ctx, backup_cand)
    if url:
        _record_choice(ctx, backup_cand, backup=True)
    return url


async def try_claim_pool(
    ctx: Context, pool: list[Candidate], seg_index: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    for i, cand in enumerate(pool):
        key = (cand.site, cand.cand_id)
        async with ctx.used_files_lock:
            if key in ctx.used_files:
                continue
            ctx.used_files.add(key)
        final_url = await finalize_candidate(ctx, cand)
        if final_url:
            if seg_index is not None:
                ctx.primary_cands[seg_index] = cand
            backup_url = await try_claim_backup(ctx, pool[i + 1:], cand.site)
            return final_url, backup_url
    return None, None


ARCHIVE_SITES = ("wikimedia", "loc", "nasa")
STOCK_SITES = ("pexels", "pixabay")


@dataclass
class Variant:
    name: str  # "narrow" | "medium" | "broad"
    query: str
    sites: list[str]
    apply_entity_filter: bool


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())


def _pixabay_first(sites: list[str]) -> list[str]:
    out: list[str] = []
    for x in sites:
        x = x.strip().lower()
        if x and x not in out:
            out.append(x)
    if "pixabay" in out and "pexels" in out and out.index("pixabay") > out.index("pexels"):
        out.remove("pixabay")
        out.insert(out.index("pexels"), "pixabay")
    return out


def entity_filter_applies(seg: SegmentSpec, variant_name: str, site: str) -> bool:
    """Фильтр сущностей: только narrow/medium и только архивные сайты.
    (seg.is_entity и skip_keyword_filter проверяются в fetch_and_filter.)"""
    return variant_name in ("narrow", "medium") and site in ARCHIVE_SITES


def build_cascade(seg: SegmentSpec) -> list[Variant]:
    """Чистая функция (без сети и ctx): упорядоченный список вариантов запроса."""
    seg_sites = [x for x in _pixabay_first(list(seg.sites)) if x in SITE_SEARCH_FUNCS]
    archive_first = bool(seg_sites) and seg_sites[0] in ARCHIVE_SITES

    if archive_first:
        order = ("narrow", "medium", "broad")
        narrow_medium_sites = [x for x in seg_sites if x in ARCHIVE_SITES]
    else:
        order = ("medium", "narrow", "broad")
        narrow_medium_sites = seg_sites
    queries = {
        "narrow": seg.query_narrow, "medium": seg.query_medium, "broad": seg.query_broad,
    }
    broad_sites = _pixabay_first(list(STOCK_SITES))

    cascade: list[Variant] = []
    seen: list[tuple[str, set]] = []
    for name in order:
        q = queries[name]
        if not q or not q.strip():
            continue
        sites = broad_sites if name == "broad" else list(narrow_medium_sites)
        if not sites:
            continue
        nq = _norm_query(q)
        if any(nq == sq and set(sites) & ss for sq, ss in seen):
            continue
        seen.append((nq, set(sites)))
        cascade.append(Variant(
            name=name, query=q.strip(), sites=sites,
            apply_entity_filter=any(entity_filter_applies(seg, name, x) for x in sites),
        ))
    return cascade


def _stats_key(site: str, variant_name: str) -> str:
    return f"{site}_broad" if variant_name == "broad" else site


async def fetch_and_filter(
    ctx: Context, site: str, seg: SegmentSpec, query: str, variant_name: str,
    stats_key: Optional[str] = None,
) -> list[Candidate]:
    q = query
    stats = ctx.site_stats.setdefault(stats_key or _stats_key(site, variant_name), SiteStats())
    stats.segments_attempted += 1

    raw = await SITE_SEARCH_FUNCS[site](ctx, q, seg.type)
    stats.raw_total += len(raw)
    if not raw:
        logging.info(
            "Сегмент %s/%s [%s]: 0 сырых кандидатов по запросу %r - сайт ничего не вернул.",
            seg.index, site, variant_name, q,
        )
        return []

    licensed = [c for c in raw if c.license_ok]
    stats.license_ok_total += len(licensed)
    if not licensed:
        logging.info(
            "Сегмент %s/%s [%s]: %s сырых кандидатов, но 0 прошло лицензионный фильтр.",
            seg.index, site, variant_name, len(raw),
        )
        return []

    apply_entity = entity_filter_applies(seg, variant_name, site)
    skip_keyword_filter = site == "pexels" and seg.type == "video"
    if seg.is_entity and apply_entity and not skip_keyword_filter:
        before = len(licensed)
        licensed = [c for c in licensed if text_matches_keywords(c.text, seg.entity_keywords)]
        if not licensed:
            logging.info(
                "Сегмент %s/%s [%s]: %s кандидатов прошли лицензию, но 0 после фильтра сущностей %r.",
                seg.index, site, variant_name, before, seg.entity_keywords,
            )
            return []
    stats.keyword_ok_total += len(licensed)
    return licensed


def best_effort_order(pool: list[Candidate]) -> list[Candidate]:
    """Порядок для режима "лучший из найденного": сначала не "чужие", затем "чужие";
    внутри групп по убыванию own_sim."""
    neutral = [c for c in pool if c.reject_reason != "foreign"]
    foreign = [c for c in pool if c.reject_reason == "foreign"]
    key = lambda c: c.own_sim
    return sorted(neutral, key=key, reverse=True) + sorted(foreign, key=key, reverse=True)


def _choice_key(cand: Candidate) -> str:
    return cand.reject_reason if cand.accepted else f"best_effort_{cand.reject_reason}"


def _record_choice(ctx: Context, cand: Candidate, seg_index: Optional[int] = None, backup: bool = False) -> None:
    key = _choice_key(cand)
    if backup:
        ctx.backup_reasons[key] += 1
        ctx.backup_own_sims.append(cand.own_sim)
        return
    ctx.choice_reasons[key] += 1
    ctx.choice_own_sims.append(cand.own_sim)
    logging.info(
        "Сегмент %s: выбран %s/%s [%s] own_sim=%.3f rank=%s причина=%s (%s)",
        seg_index, cand.site, cand.cand_id, cand.variant, cand.own_sim, cand.rank,
        cand.reject_reason, "принят" if cand.accepted else "best-effort",
    )


def summarize_choices(reasons, own_sims: list, label: str = "primary") -> str:
    g = lambda k: int(reasons.get(k, 0))
    be = {k: g(f"best_effort_{k}") for k in ("floor", "foreign", "rank_miss", "flat")}
    total = sum(int(v) for v in reasons.values())
    line = (
        f"Итог выбора {label}: всего {total}; по рангу {g('rank')}; по абсолютному порогу {g('abs')}; "
        f"best-effort {sum(be.values())} (floor {be['floor']}, foreign {be['foreign']}, "
        f"rank_miss {be['rank_miss']}, flat {be['flat']})"
    )
    if own_sims:
        line += (
            f"; own_sim выбранных: min {min(own_sims):.3f}, "
            f"median {statistics.median(own_sims):.3f}, max {max(own_sims):.3f}"
        )
    return line


def select_accepted(seg: SegmentSpec, scored: list[Candidate]) -> list[Candidate]:
    """Единая точка решения: кого принять из оценённых. Сейчас - принятые
    относительной оценкой (Candidate.accepted), по убыванию own_sim."""
    return sorted((c for c in scored if c.accepted), key=lambda c: c.own_sim, reverse=True)


async def run_variant(ctx: Context, seg: SegmentSpec, variant: Variant) -> list[Candidate]:
    """Оценивает вариант; возвращает ВСЕХ оценённых (принятых и нет), по убыванию own_sim."""
    pool: list[Candidate] = []
    for site in variant.sites:
        if site in ctx.exhausted_sites or site not in SITE_SEARCH_FUNCS:
            continue
        key = _stats_key(site, variant.name)
        licensed = await fetch_and_filter(ctx, site, seg, variant.query, variant.name, stats_key=key)
        if not licensed:
            continue
        top = licensed[:CANDIDATES_PER_SITE]
        scored = await score_candidates(
            ctx, top, seg_index=seg.index, stats_key=key, variant_name=variant.name,
        )
        if not scored:
            continue
        pool.extend(scored)
        pool.sort(key=lambda c: c.own_sim, reverse=True)
        if any(c.accepted for c in scored):
            break
    return pool


async def process_segment_inner(ctx: Context, seg: SegmentSpec) -> tuple[Optional[str], Optional[str]]:
    rejected_pool: list[Candidate] = []
    for variant in build_cascade(seg):
        scored = await run_variant(ctx, seg, variant)
        accepted = select_accepted(seg, scored)
        if accepted:
            url, backup_url = await try_claim_pool(ctx, accepted, seg.index)
            if url:
                chosen = ctx.primary_cands.get(seg.index)
                if chosen is not None:
                    _record_choice(ctx, chosen, seg.index)
                return url, backup_url
        rejected_pool.extend(c for c in scored if not c.accepted)

    # Никто не принят (или принятых не удалось заклеймить): берём лучшего по own_sim.
    if rejected_pool:
        url, backup_url = await try_claim_pool(ctx, best_effort_order(rejected_pool), seg.index)
        if url:
            chosen = ctx.primary_cands.get(seg.index)
            if chosen is not None:
                ctx.site_stats.setdefault(chosen.stats_key or chosen.site, SiteStats()).best_effort_total += 1
                _record_choice(ctx, chosen, seg.index)
            return url, backup_url
    return None, None


async def process_segment(ctx: Context, seg: SegmentSpec) -> tuple[int, Optional[str], Optional[str]]:
    async with ctx.global_semaphore:
        url, backup_url = await process_segment_inner(ctx, seg)
    return seg.index, url, backup_url


async def _backup_pool(
    ctx: Context, site: str, seg: SegmentSpec, variant: Variant, exclude_key: tuple,
) -> list[Candidate]:
    licensed = await fetch_and_filter(
        ctx, site, seg, variant.query, variant.name, stats_key=f"{site}_backup",
    )
    if not licensed:
        return []
    async with ctx.used_files_lock:
        fresh = [
            c for c in licensed
            if (c.site, c.cand_id) not in ctx.used_files
            and (c.site, c.cand_id) != exclude_key
        ]
    top = fresh[:CANDIDATES_PER_SITE]
    if not top:
        return []
    return await score_candidates(
        ctx, top, seg_index=seg.index, stats_key=f"{site}_backup", variant_name=variant.name,
    )


async def _claim_first(
    ctx: Context, pool: list[Candidate],
) -> tuple[Optional[str], Optional[Candidate]]:
    """pool уже упорядочен (select_accepted или по own_sim)."""
    for cand in pool:
        key = (cand.site, cand.cand_id)
        async with ctx.used_files_lock:
            if key in ctx.used_files:
                continue
            ctx.used_files.add(key)
        url = await finalize_candidate(ctx, cand)
        if url:
            return url, cand
    return None, None


async def find_backup(ctx: Context, seg: SegmentSpec) -> tuple[Optional[str], Optional[str]]:
    """B1 (другие сайты) -> B2 (тот же сайт). Возвращает (url, 'other'|'same') или (None, None)."""
    primary = ctx.primary_cands.get(seg.index)
    if primary is None:
        return None, None
    pkey = (primary.site, primary.cand_id)

    cascade = build_cascade(seg)

    candidate_sites = list(seg.sites) + BACKUP_EXTRA_SITES
    # Если primary был loc, обязательно добавляем pexels и pixabay в список резерва
    if primary.site == "loc":
        for stock_site in ("pexels", "pixabay"):
            if stock_site not in candidate_sites:
                candidate_sites.append(stock_site)

    ordered: list[str] = []
    for site in candidate_sites:
        if site in SITE_SEARCH_FUNCS and site != primary.site and site not in ordered:
            ordered.append(site)
    groups = ([x for x in ordered if x != "loc"], [x for x in ordered if x == "loc"])

    rejected: list[Candidate] = []  # оценённые, но не принятые: запас "лучший из найденного"

    # B1: поиск по другим сайтам
    for group in groups:
        if not group:
            continue
        for variant in cascade:
            pool: list[Candidate] = []
            for site in (x for x in variant.sites if x in group):
                if site in ctx.exhausted_sites:
                    continue
                scored = await _backup_pool(ctx, site, seg, variant, pkey)
                pool.extend(scored)
                if any(c.accepted for c in scored):
                    break
            if pool:
                url, _c = await _claim_first(ctx, select_accepted(seg, pool))
                if url:
                    if _c is not None:
                        _record_choice(ctx, _c, backup=True)
                    return url, "other"
                rejected.extend(c for c in pool if not c.accepted)

    # B2: тот же сайт (для LOC полностью запрещено, чтобы не создать двойной отказ)
    if primary.site not in ctx.exhausted_sites and primary.site != "loc":
        for variant in cascade:
            if primary.site not in variant.sites:
                continue
            pool = await _backup_pool(ctx, primary.site, seg, variant, pkey)
            if pool:
                url, _c = await _claim_first(ctx, select_accepted(seg, pool))
                if url:
                    if _c is not None:
                        _record_choice(ctx, _c, backup=True)
                    return url, "same"
                rejected.extend(c for c in pool if not c.accepted)

    # Принятых бэкапов нет: бэкап лучше иметь, чем нет - берём лучшего по own_sim.
    if rejected:
        url, cand = await _claim_first(ctx, best_effort_order(rejected))
        if url and cand is not None:
            _record_choice(ctx, cand, backup=True)
            ctx.site_stats.setdefault(cand.stats_key or cand.site, SiteStats()).best_effort_total += 1
            logging.info(
                "Сегмент %s: принятых backup нет, выбран лучший без принятия: %s/%s (own_sim=%.3f).",
                seg.index, cand.site, cand.cand_id, cand.own_sim,
            )
            return url, ("same" if cand.site == primary.site else "other")
    return None, None


async def run_backup_pass(
    ctx: Context, segments: list[SegmentSpec], results: dict, backups: dict,
) -> list[int]:
    t0 = time.monotonic()
    by_idx = {sg.index: sg for sg in segments}
    todo = [by_idx[i] for i in sorted(results) if i not in backups and i in by_idx]
    n_tail = len(backups)

    async def one(seg: SegmentSpec):
        try:
            async with ctx.global_semaphore:
                url, src = await find_backup(ctx, seg)
        except Exception as e:
            logging.warning("Сегмент %s: ошибка поиска backup: %s", seg.index, e)
            return seg.index, None, None
        return seg.index, url, src

    n_other = n_same = 0
    still: list[int] = []
    for coro in asyncio.as_completed([one(sg) for sg in todo]):
        idx, url, src = await coro
        if url:
            backups[idx] = url
            if src == "other":
                n_other += 1
            else:
                n_same += 1
        else:
            still.append(idx)
            logging.warning("Сегмент %s: backup не найден (B1/B2 пусты)", idx)
    still.sort()
    logging.info(
        "Backup: из хвоста %s, из другого сайта %s, из того же сайта %s, не найден %s%s. "
        "Проход занял %.1fs",
        n_tail, n_other, n_same, len(still),
        f" (сегменты: {', '.join(map(str, still))})" if still else "",
        time.monotonic() - t0,
    )
    return still


# ---------------------------------------------------------------------------
# Оркестрация запуска
# ---------------------------------------------------------------------------

async def run_search(ctx: Context, segments: list[SegmentSpec]) -> tuple[dict, dict, list]:
    total = len(segments)
    done_count = 0
    results: dict = {}
    backups: dict = {}
    missing: list = []

    async def wrapped(seg: SegmentSpec):
        nonlocal done_count
        idx, url, backup_url = await process_segment(ctx, seg)
        done_count += 1
        pct = done_count / total * 100
        logging.info(
            "Сегмент %s обработан (%s/%s, %.0f%%): %s%s",
            idx, done_count, total, pct,
            "найдено" if url else "НЕ найдено",
            " (+backup)" if backup_url else "",
        )
        return idx, url, backup_url

    tasks = [asyncio.ensure_future(wrapped(seg)) for seg in segments]
    try:
        for coro in asyncio.as_completed(tasks):
            idx, url, backup_url = await coro
            if url:
                results[idx] = url
                if backup_url:
                    backups[idx] = backup_url
            else:
                missing.append(idx)
    except FatalConfigError:
        for t in tasks:
            t.cancel()
        raise

    return results, backups, missing


def load_requests(path: str) -> list[SegmentSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("requests.json пуст или имеет неверную структуру (ожидался объект-словарь)")

    def bad(k, field, what):
        return ValueError(f"Сегмент {k!r} в requests.json: поле {field!r} {what}")

    specs: list[SegmentSpec] = []
    for k, v in data.items():
        try:
            idx = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Ключ сегмента {k!r} в requests.json не является номером") from e
        if not isinstance(v, dict):
            raise ValueError(f"Сегмент {k!r} в requests.json: ожидался объект")
        if "query" in v or any(f not in v for f in ("scene", "query_narrow", "query_medium")):
            raise ValueError(
                f"Сегмент {k!r}: requests.json старого формата, "
                "перегенерируйте его generate_queries.py"
            )
        strs = {}
        for field in ("scene", "query_narrow", "query_medium"):
            val = v[field]
            if not isinstance(val, str) or not val.strip():
                raise bad(k, field, "должно быть непустой строкой")
            strs[field] = val.strip()
        broad = v.get("query_broad")
        if broad is not None and not isinstance(broad, str):
            raise bad(k, "query_broad", "должно быть строкой или null")
        broad = broad.strip() if isinstance(broad, str) else None
        broad = broad or None
        sites = v.get("sites")
        if not isinstance(sites, list) or not sites or not all(isinstance(x, str) for x in sites):
            raise bad(k, "sites", "должно быть непустым списком строк")
        raw_type = v.get("type")
        if not isinstance(raw_type, str):
            raise bad(k, "type", "должно быть строкой \"image\" или \"video\"")
        seg_type = raw_type.strip().lower()
        if seg_type not in ("image", "video"):
            raise bad(k, "type", f"должно быть \"image\" или \"video\", получено {raw_type!r}")
        norm_sites = [x.strip().lower() for x in sites]
        unknown = [x for x in norm_sites if x not in SITE_SEARCH_FUNCS]
        if unknown:
            logging.warning(
                "Сегмент %s: неизвестные сайты в sites %s - они будут проигнорированы.", k, unknown,
            )
        try:
            specs.append(SegmentSpec(
                index=idx,
                scene=strs["scene"],
                sites=norm_sites,
                query_narrow=strs["query_narrow"],
                query_medium=strs["query_medium"],
                query_broad=broad,
                type=seg_type,
                is_entity=bool(v["is_entity"]),
                entity_keywords=list(v.get("entity_keywords") or []),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Сегмент {k!r} в requests.json имеет некорректную структуру: {e}") from e

    specs.sort(key=lambda s: s.index)
    return specs


async def amain(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.input):
        logging.error("Входной файл не найден: %s", args.input)
        return 1

    try:
        segments = load_requests(args.input)
    except (ValueError, json.JSONDecodeError) as e:
        logging.error("Ошибка чтения %s: %s", args.input, e)
        return 1

    logging.info("Загружено сегментов: %s", len(segments))
    rel_top_n = effective_top_n(len(segments))
    logging.info(
        "Относительная оценка CLIP: SIM_MIN_THRESHOLD(floor)=%.3f, SIM_ACCEPT_THRESHOLD(abs)=%.3f, "
        "REL_TOP_N=%s (%s, эффективное для %s сегментов: %s), REL_MARGIN=%.3f, FLAT_SPREAD=%.3f "
        "(SEARCH_SIM_MIN_THRESHOLD / SEARCH_SIM_ACCEPT_THRESHOLD / SEARCH_REL_TOP_N / "
        "SEARCH_REL_MARGIN / SEARCH_FLAT_SPREAD).",
        SIM_MIN_THRESHOLD, SIM_ACCEPT_THRESHOLD, REL_TOP_N,
        "задано явно" if REL_TOP_N_EXPLICIT else "потолок по умолчанию",
        len(segments), rel_top_n, REL_MARGIN, FLAT_SPREAD,
    )

    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    pixabay_key = os.environ.get("PIXABAY_API_KEY", "")
    if not pexels_key or not pixabay_key:
        logging.error(
            "PEXELS_API_KEY и/или PIXABAY_API_KEY не заданы - без них поиск невозможен "
            "(в т.ч. broad-вариант каскада всегда идёт через pixabay/pexels)."
        )
        return 1

    if LOC_MIN_INTERVAL_SECONDS > 0:
        logging.info(
            "LOC rate-limiter активен: минимум %.2fs между последовательными запросами "
            "метаданных/поиска (%.1f запросов/мин; официальный лимит JSON/YAML API LOC - "
            "20/мин с часовой блокировкой при превышении). Настраивается через "
            "SEARCH_LOC_MIN_INTERVAL_SECONDS. При первом же 429 (или CAPTCHA-ответе) сайт "
            "LOC помечается исчерпанным на весь остаток запуска - повторные попытки в "
            "рамках часовой блокировки не имеют смысла.",
            LOC_MIN_INTERVAL_SECONDS, 60.0 / LOC_MIN_INTERVAL_SECONDS,
        )
    if LOC_PREVIEW_MIN_INTERVAL_SECONDS > 0:
        logging.info(
            "LOC-preview rate-limiter активен: минимум %.2fs между запросами превью-файлов "
            "(%.1f запросов/мин; официальный лимит Media content /storage-services/ у LOC - "
            "150/мин). Настраивается через SEARCH_LOC_PREVIEW_MIN_INTERVAL_SECONDS. Отдельный "
            "от лимитера метаданных выше - разные эндпоинты, разные официальные лимиты.",
            LOC_PREVIEW_MIN_INTERVAL_SECONDS, 60.0 / LOC_PREVIEW_MIN_INTERVAL_SECONDS,
        )
    if WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS > 0:
        logging.info(
            "Wikimedia-preview rate-limiter активен: минимум %.2fs между превью "
            "(конкурентность: %d). Настраивается через SEARCH_WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS / "
            "SEARCH_WIKIMEDIA_PREVIEW_CONCURRENCY. Защищает CDN от 429 при высокой сегментной параллельности.",
            WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS, WIKIMEDIA_PREVIEW_CONCURRENCY,
        )

    site_semaphores = {s: asyncio.Semaphore(v) for s, v in SEMAPHORE_DEFAULTS.items()}
    site_semaphores["wikimedia_preview"] = asyncio.Semaphore(WIKIMEDIA_PREVIEW_CONCURRENCY)

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        connector=connector, headers={"User-Agent": SESSION_USER_AGENT},
    ) as session:
        ctx = Context(
            session=session,
            pexels_api_key=pexels_key,
            pixabay_api_key=pixabay_key,
            site_semaphores=site_semaphores,
            global_semaphore=asyncio.Semaphore(GLOBAL_SEGMENT_CONCURRENCY),
            clip_semaphore=asyncio.Semaphore(CLIP_CONCURRENCY),
            used_files_lock=asyncio.Lock(),
            rate_limiters={
                "loc": RateLimiter(LOC_MIN_INTERVAL_SECONDS),
                "loc_preview": RateLimiter(LOC_PREVIEW_MIN_INTERVAL_SECONDS),
                "wikimedia_preview": RateLimiter(WIKIMEDIA_PREVIEW_MIN_INTERVAL_SECONDS),
            },
        )
        ctx.clip = ClipScorer(CLIP_MODEL_NAME, CLIP_PRETRAINED)
        ctx.rel_top_n = rel_top_n
        try:
            t0 = time.monotonic()
            await ctx.clip.ensure_loaded()
            t_load = time.monotonic() - t0
            t1 = time.monotonic()
            await ctx.clip.encode_scenes([(sg.index, sg.scene) for sg in segments])
            logging.info(
                "CLIP: загрузка модели %.1fs, кодирование scene (%s шт.) %.1fs.",
                t_load, len(segments), time.monotonic() - t1,
            )
        except FatalConfigError:
            raise
        except Exception as e:
            logging.error(
                "Не удалось загрузить CLIP (%s / %s) или закодировать scene: %s. Проверьте "
                "SEARCH_CLIP_MODEL / SEARCH_CLIP_PRETRAINED, доступ к интернету для первой загрузки "
                "весов и кэш actions/cache.",
                CLIP_MODEL_NAME, CLIP_PRETRAINED, e,
            )
            return 1

        try:
            results, backups, missing = await run_search(ctx, segments)
        except FatalConfigError as e:
            logging.error("Структурная ошибка конфигурации: %s", e)
            log_site_stats_summary(ctx.site_stats)
            return 1

        backup_missing = await run_backup_pass(ctx, segments, results, backups)

        logging.info(summarize_choices(ctx.choice_reasons, ctx.choice_own_sims, "primary"))
        logging.info(summarize_choices(ctx.backup_reasons, ctx.backup_own_sims, "backup"))

        log_site_stats_summary(ctx.site_stats)

    with open(args.links_output, "w", encoding="utf-8") as f:
        for idx in sorted(results):
            f.write(f"{idx}: {results[idx]}\n")

    with open(args.backup_links_output, "w", encoding="utf-8") as f:
        for idx in sorted(backups):
            f.write(f"{idx}: {backups[idx]}\n")

    with open(args.missing_output, "w", encoding="utf-8") as f:
        for idx in sorted(missing):
            f.write(f"{idx}\n")

    with open(args.backup_missing_output, "w", encoding="utf-8") as f:
        for idx in backup_missing:
            f.write(f"{idx}\n")

    reject_line = ctx.format_rejects.summary_line()
    if reject_line:
        logging.info(reject_line)

    logging.info(
        "Готово: найдено %s из %s сегментов (из них с backup - %s), не найдено %s, "
        "backup_missing %s. Результаты: %s, backup: %s, пропуски: %s, без backup: %s",
        len(results), len(segments), len(backups), len(missing), len(backup_missing),
        args.links_output, args.backup_links_output, args.missing_output,
        args.backup_missing_output,
    )
    return 0


def _selftest() -> int:
    import tempfile

    def mk(sites, n="wiki narrow", m="wiki medium", b="city street", **kw):
        return SegmentSpec(index=1, scene="s", sites=sites, query_narrow=n, query_medium=m,
                           query_broad=b, type="image", is_entity=True, entity_keywords=["x"])

    def view(seg):
        return [(v.name, v.sites) for v in build_cascade(seg)]

    # архивный первый
    assert view(mk(["wikimedia", "pexels"])) == [
        ("narrow", ["wikimedia"]), ("medium", ["wikimedia"]), ("broad", ["pixabay", "pexels"])]
    # сток первый
    assert view(mk(["pexels", "wikimedia"])) == [
        ("medium", ["pexels", "wikimedia"]), ("narrow", ["pexels", "wikimedia"]),
        ("broad", ["pixabay", "pexels"])]
    c = view(mk(["pexels", "pixabay"]))
    assert [n for n, _ in c] == ["medium", "narrow", "broad"], c
    assert c[0][1] == ["pixabay", "pexels"], c  # Pixabay перед Pexels
    # смешанный
    assert view(mk(["wikimedia", "loc", "pexels", "pixabay"])) == [
        ("narrow", ["wikimedia", "loc"]), ("medium", ["wikimedia", "loc"]),
        ("broad", ["pixabay", "pexels"])]
    # дубли narrow == medium (регистр/пробелы)
    c = view(mk(["wikimedia"], n="Hagia  Sophia", m="hagia sophia"))
    assert [n for n, _ in c] == ["narrow", "broad"], c
    # broad=None
    assert [n for n, _ in view(mk(["wikimedia"], b=None))] == ["narrow", "medium"]
    # broad совпал с medium, но сайты не пересекаются -> не отбрасывается
    assert [n for n, _ in view(mk(["wikimedia"], n="a b", m="c d", b="C D"))] == [
        "narrow", "medium", "broad"]
    # фильтр сущностей
    seg = mk(["wikimedia"])
    assert entity_filter_applies(seg, "narrow", "wikimedia") is True
    assert entity_filter_applies(seg, "medium", "loc") is True
    assert entity_filter_applies(seg, "broad", "pixabay") is False
    assert entity_filter_applies(seg, "medium", "pexels") is False
    assert entity_filter_applies(seg, "broad", "wikimedia") is False
    # неизвестные сайты
    c = view(mk(["flickr", "pexels"]))
    assert all("flickr" not in st for _, st in c), c
    assert view(mk(["flickr", "wikimedia", "pexels"])) == [
        ("narrow", ["wikimedia"]), ("medium", ["wikimedia"]), ("broad", ["pixabay", "pexels"])]
    assert view(mk(["flickr", "foo"])) == [("broad", ["pixabay", "pexels"])]
    # load_requests
    good = {"1": {"scene": "s", "sites": ["Pexels"], "query_narrow": "a b c", "query_medium": "a b",
                  "query_broad": "", "type": "video", "is_entity": False, "entity_keywords": []}}
    old = {"1": {"sites": ["pexels"], "query": "a", "type": "image",
                 "is_entity": False}}
    badtype = {"2": dict(good["1"], query_medium="  ")}

    def load(obj):
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "requests.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(obj, f)
            return load_requests(fp)

    specs = load(good)
    assert specs[0].query_broad is None and specs[0].sites == ["pexels"]
    notype = {"3": {k2: v2 for k2, v2 in good["1"].items() if k2 != "type"}}
    gif = {"4": dict(good["1"], type="gif")}
    assert load({"1": dict(good["1"], type="Video", sites=["flickr", "pexels"])})[0].type == "video"
    for obj, needle in ((old, "старого формата"), (badtype, "query_medium"),
                        (notype, "'type'"), (gif, "'type'")):
        try:
            load(obj)
        except ValueError as e:
            assert needle in str(e) and "Сегмент" in str(e), e
        else:
            raise AssertionError("ожидалась ValueError")
    # --- относительная оценка (без CLIP, на искусственных векторах) ---
    import numpy as np
    kw = dict(top_n=2, margin=0.03, min_abs=0.21, flat_spread=0.03, accept_abs=0.30)
    # своя сцена явно лучшая -> принят по рангу
    r = rank_decision([0.28, 0.22, 0.20, 0.15, 0.12], 0, **kw)
    assert (r.accepted, r.rank, r.reason) == (True, 1, "rank"), r
    assert rank_candidate([0.28, 0.22, 0.20, 0.15, 0.12], 0, **kw) == (True, 0.28, 1)
    # чужая сцена выше на margin -> чужой, даже при ранге 2 <= top_n
    r = rank_decision([0.24, 0.30 - 0.0001, 0.15, 0.12], 0, **kw)
    assert (r.accepted, r.rank, r.reason) == (False, 2, "foreign"), r
    # плоское распределение: решает абсолютный критерий
    assert rank_decision([0.25, 0.26, 0.255, 0.25], 0, **kw).accepted is False
    assert rank_decision([0.31, 0.305, 0.30, 0.31], 0, **kw).reason == "abs"
    # ниже нижней страховки -> не принят
    r = rank_decision([0.20, 0.10, 0.05], 0, **kw)
    assert (r.accepted, r.reason) == (False, "floor"), r
    # абсолютный критерий не блокируется "чужим"
    assert rank_decision([0.31, 0.35, 0.34, 0.33, 0.32], 0, **kw).reason == "abs"
    # вне топ-N и не чужой по margin -> rank_miss
    assert rank_decision([0.22, 0.23, 0.235, 0.24, 0.245, 0.10], 0, **kw).accepted is False
    # матрица сходств на искусственных нормализованных векторах
    m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.6, 0.8, 0, 0]], dtype=np.float32)
    v = np.array([0.6, 0.8, 0, 0], dtype=np.float32)
    sims = compute_sims(m, v)
    assert np.allclose(sims, [0.6, 0.8, 1.0], atol=1e-6), sims
    # эффективное N
    assert effective_top_n(50, 10, False) == 3
    assert effective_top_n(126, 10, False) == 7
    assert effective_top_n(1000, 10, False) == 10
    assert effective_top_n(1000, 5, True) == 5
    # кэш: один ключ кодируется один раз, в т.ч. при параллельных запросах
    import asyncio as _a
    calls = {"n": 0}

    async def fake_encode():
        calls["n"] += 1
        await _a.sleep(0.01)
        return np.ones(4, dtype=np.float32)

    async def cache_test():
        c = EmbeddingCache()
        await _a.gather(*[c.get_or_compute(("pexels", "1"), fake_encode) for _ in range(5)])
        await c.get_or_compute(("pexels", "1"), fake_encode)
        assert calls["n"] == 1, calls
        await c.get_or_compute(("pexels", "2"), fake_encode)
        assert calls["n"] == 2 and len(c) == 2

        async def none_enc():
            calls["n"] += 1
            return None
        await c.get_or_compute(("x", "3"), none_enc)
        await c.get_or_compute(("x", "3"), none_enc)
        assert len(c) == 2  # None не кэшируется
    _a.run(cache_test())
    # select_accepted
    cs = [Candidate("p", str(i), "", True, None, None, own_sim=x, similarity=x, accepted=a)
          for i, (x, a) in enumerate([(0.25, True), (0.3, False), (0.28, True)])]
    assert [c.cand_id for c in select_accepted(mk(["pexels"]), cs)] == ["2", "0"]
    # best_effort_order / summarize_choices / Candidate.variant
    mkc = lambda i, x, r: Candidate("p", str(i), "", True, None, None, own_sim=x, similarity=x, reject_reason=r)
    order = best_effort_order([mkc(1, 0.26, "foreign"), mkc(2, 0.24, "rank_miss"), mkc(3, 0.22, "floor")])
    assert [(c.cand_id) for c in order] == ["2", "3", "1"], order
    assert [c.cand_id for c in best_effort_order([mkc(1, 0.2, "foreign"), mkc(2, 0.25, "foreign")])] == ["2", "1"]
    assert best_effort_order([]) == []
    assert "всего 0" in summarize_choices(Counter(), []) and "min" not in summarize_choices({}, [])
    line = summarize_choices(
        Counter({"rank": 2, "abs": 1, "best_effort_foreign": 1, "best_effort_floor": 1}),
        [0.20, 0.30, 0.25, 0.40], "primary")
    for part in ("всего 5", "по рангу 2", "по абсолютному порогу 1", "best-effort 2", "floor 1",
                 "foreign 1", "min 0.200", "median 0.275", "max 0.400"):
        assert part in line, (part, line)
    c0 = mkc(9, 0.3, "rank")
    assert c0.variant is None and dc_replace(c0, variant="narrow").variant == "narrow"
    assert dc_replace(dc_replace(c0, variant="broad"), own_sim=0.1).variant == "broad"
    print("selftest OK")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Подбор медиа для сегментов requests.json")
    parser.add_argument("--input", default=os.environ.get("SEARCH_INPUT", DEFAULT_SEARCH_INPUT))
    parser.add_argument("--links-output", default=os.environ.get("SEARCH_LINKS_OUTPUT", DEFAULT_LINKS_OUTPUT))
    parser.add_argument(
        "--backup-links-output",
        default=os.environ.get("SEARCH_BACKUP_LINKS_OUTPUT", DEFAULT_BACKUP_LINKS_OUTPUT),
    )
    parser.add_argument("--missing-output", default=os.environ.get("SEARCH_MISSING_OUTPUT", DEFAULT_MISSING_OUTPUT))
    parser.add_argument(
        "--backup-missing-output",
        default=os.environ.get("SEARCH_BACKUP_MISSING_OUTPUT", DEFAULT_BACKUP_MISSING_OUTPUT),
    )
    parser.add_argument("--selftest", action="store_true", help="тесты чистых функций без сети")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()

    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
