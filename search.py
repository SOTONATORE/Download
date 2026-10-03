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
    min_durations.json - {"номер": секунды} ТОЛЬКО для видео-сегментов (type=video, не skip);
                        лежит в том же каталоге, что и links.txt; значение = min_duration из
                        requests.json без изменений (читает download.py)

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
    MEDIA_MODE - 1 (смешанный, по умолч. при пустом/отсутствующем), 2 (только видео), 3 (только
        фото). Иное значение - остановка с сообщением. В режиме 2 не опрашиваются сайты, не
        отдающие видео из белого списка (сейчас wikimedia), каждый пропуск пишется в лог.
        Тип сегмента всё равно берётся из requests.json (seg.type).
        Только в режиме 1: если по типу из requests.json нет кандидата, принятого по abs/rank,
        каскад повторяется с ДРУГИМ типом (image <-> video) по тем же запросам; best-effort - только
        если не принят никто ни по одному из типов (см. process_segment_inner). Итоговый тип
        сегмента попадает в min_durations.json. В режимах 2 и 3 переключения нет.
    SEARCH_PIXABAY_STRICT_TYPES - типы контента в запросах к Pixabay (умолч. 1): для видео
        добавляется video_type=film, для фото image_type=photo (отсекает анимацию, 3D-рендеры,
        иллюстрации). 0/false/no/off - параметры не добавляются (прежнее поведение). Читается один раз.

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
   векторов. Превью кандидата кодируется один раз (кэш по (site, kind, cand_id), только вектор ~2 КБ);
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

Слабое/сильное принятие, вариант name, потолки кандидатов, блок-лист Wikimedia:
 - Сильное принятие (is_strong): причина "abs" либо (ранг <= 2 и own_sim >= STRONG_OWN_SIM,
   env SEARCH_STRONG_OWN_SIM, умолч. 0.24). Если принятые есть, но все слабые, каскад идёт дальше;
   после последнего варианта клеймится лучший по own_sim из всех слабых (в логе "принят, слабый").
   Ранний выход по сайтам внутри варианта - только при сильном принятом.
 - Вариант "name" (только для is_entity и архивного первого сайта): после medium, перед broad;
   запрос - первый латинский элемент entity_keywords; сайты и фильтр сущностей как у medium;
   ключ статистики - имя сайта; дедупликация прежняя. Имена вариантов: narrow, medium, name, broad.
 - Потолки кандидатов под CLIP: SEARCH_CANDIDATES_STOCK (15; pexels, pixabay) и
   SEARCH_CANDIDATES_ARCHIVE (8; wikimedia, loc, nasa). Явно заданный SEARCH_CANDIDATES_PER_SITE
   перекрывает оба для всех сайтов (candidates_cap).
 - Блок-лист Wikimedia (WIKIMEDIA_BLOCKLIST_WORDS; env SEARCH_WIKIMEDIA_BLOCKLIST - слова через
   запятую, пустая строка отключает): кандидат отсеивается до CLIP, если в его тексте есть слово
   блок-листа, которого нет в scene (wikimedia_blocked); счётчик - blocked_total в сводке по воронке.
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

# Потолки кандидатов под CLIP: сток дешёвый (превью без лимитеров), архивы дороги (лимитеры превью).
# SEARCH_CANDIDATES_PER_SITE, если задан явно, перекрывает оба потолка для всех сайтов.
_CAND_PER_SITE_ENV = os.environ.get("SEARCH_CANDIDATES_PER_SITE")
CANDIDATES_OVERRIDE: Optional[int] = int(_CAND_PER_SITE_ENV) if _CAND_PER_SITE_ENV else None
CANDIDATES_STOCK = int(os.environ.get("SEARCH_CANDIDATES_STOCK", 15))
CANDIDATES_ARCHIVE = int(os.environ.get("SEARCH_CANDIDATES_ARCHIVE", 8))

# "Сильное" принятие: см. is_strong.
STRONG_OWN_SIM = float(os.environ.get("SEARCH_STRONG_OWN_SIM", 0.24))

# Блок-лист типов контента Wikimedia (марки, карты, сканы...). Пустая строка в env - отключить.
_DEFAULT_WM_BLOCKLIST = (
    "stamp", "stamps", "coin", "coins", "banknote", "banknotes", "map", "maps", "poster",
    "logo", "flag", "flags", "coat of arms", "diagram", "newspaper", "book cover",
    "title page", "page",
    "calendar", "document", "chart", "infographic", "emblem", "id card", "passport",
)
# Общий список "мусорных типов кадра": для этих слов исключение "слово есть в scene" в
# wikimedia_blocked НЕ действует (блокируются всегда, если слово есть в блок-листе).
# Множественное число учитывается формами w / w+s / w без s. SEARCH_WIKIMEDIA_STRICT_BLOCK=0 -
# вернуть прежнее поведение (исключение по сцене для всех слов).
WIKIMEDIA_STRICT_BLOCK_WORDS: tuple = (
    "map", "calendar", "document", "chart", "diagram", "infographic", "flag", "emblem",
    "coat of arms", "newspaper", "ID card", "passport",
)
WIKIMEDIA_STRICT_BLOCK: bool = (
    os.environ.get("SEARCH_WIKIMEDIA_STRICT_BLOCK", "1").strip().lower() not in ("0", "false", "no", "off")
)

# Pixabay: ограничение типа контента (видео -> video_type=film, фото -> image_type=photo).
# SEARCH_PIXABAY_STRICT_TYPES=0 - параметры не добавляются (прежнее поведение).
PIXABAY_STRICT_TYPES: bool = (
    os.environ.get("SEARCH_PIXABAY_STRICT_TYPES", "1").strip().lower() not in ("0", "false", "no", "off")
)

# Режим источников: 1 = только архивные, 2 = микс (по умолчанию), 3 = только стоки.
# Любое другое значение игнорируется (WARNING в amain) и берётся 2.
SOURCES_MODE_DEFAULT = 2
_SOURCES_MODE_RAW = os.environ.get("SEARCH_SOURCES_MODE")


def parse_sources_mode(raw: Optional[str]) -> tuple[int, bool]:
    """(режим, значение_корректно). Пустое/не заданное -> (2, True); мусор -> (2, False)."""
    if raw is None or not raw.strip():
        return SOURCES_MODE_DEFAULT, True
    v = raw.strip()
    if v in ("1", "2", "3"):
        return int(v), True
    return SOURCES_MODE_DEFAULT, False


SOURCES_MODE, _SOURCES_MODE_VALID = parse_sources_mode(_SOURCES_MODE_RAW)

# Режим медиа (env MEDIA_MODE): 1 = смешанный, 2 = только видео, 3 = только фото.
# Читается ОДИН раз в amain (parse_media_mode) и дальше лежит в Context.media_mode.
MEDIA_MODE_DEFAULT = 1
MEDIA_MODE_NAMES = {1: "смешанный", 2: "только видео", 3: "только фото"}


def parse_media_mode(raw: Optional[str]) -> int:
    """Чистая функция. None/пусто -> 1; "1"/"2"/"3" (пробелы по краям игнорируются) -> число;
    любое другое значение -> ValueError с сообщением по-русски (вызывающий останавливает работу)."""
    if raw is None or not raw.strip():
        return MEDIA_MODE_DEFAULT
    v = raw.strip()
    if v in ("1", "2", "3"):
        return int(v)
    raise ValueError(
        f"MEDIA_MODE={raw!r} недопустим: ожидается 1 (смешанный), 2 (только видео), "
        "3 (только фото) или пустое значение (= 1)."
    )


# Ранний фильтр по длительности видео (до CLIP). Мягкий: отсеивается только то, что ЯВНО короче
# min_duration - DURATION_TOLERANCE_SECONDS; строгая проверка по реальному файлу - в download.py.
DURATION_TOLERANCE_SECONDS = 1.0
DURATION_FILTER_SITES = ("pexels", "pixabay")  # только у них duration есть в ответе поиска


def parse_duration_seconds(raw: Any) -> Optional[float]:
    """Чистая функция. Секунды как float или None, если значение неизвестно/мусор: нет поля,
    не число (в т.ч. строка и bool), NaN/inf, ноль или отрицательное."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    d = float(raw)
    if not math.isfinite(d) or d <= 0:
        return None
    return d


def duration_verdict(raw: Any, min_duration: float) -> str:
    """Чистая функция. "reject" - явно короче (d < min_duration - 1.0); "keep" - подходит или
    попадает в зону неопределённости (граница включительно); "unknown" - длительность неизвестна
    (кандидат НЕ отсеивается)."""
    d = parse_duration_seconds(raw)
    if d is None:
        return "unknown"
    if d < float(min_duration) - DURATION_TOLERANCE_SECONDS:
        return "reject"
    return "keep"


MIN_DURATIONS_FILENAME = "min_durations.json"


def build_min_durations(segments: Sequence["SegmentSpec"]) -> dict:
    """Чистая функция. {"номер": float} только для type == "video" и не skip, по возрастанию
    номера; значение min_duration без изменений. Видео без min_duration -> ValueError."""
    out: dict = {}
    for sg in sorted(segments, key=lambda x: x.index):
        if sg.type != "video" or sg.skip:
            continue
        if sg.min_duration is None:
            raise ValueError(
                f"Сегмент {sg.index}: type=video, но в requests.json нет min_duration - "
                "min_durations.json не может быть записан (подстановка значения не делается)."
            )
        out[str(sg.index)] = float(sg.min_duration)
    return out


def min_durations_path(links_output: str) -> str:
    """min_durations.json кладётся в тот же каталог, что и links.txt."""
    return os.path.join(os.path.dirname(os.path.abspath(links_output)), MIN_DURATIONS_FILENAME)


def write_json_atomic(path: str, obj: Any) -> None:
    """UTF-8, временный файл рядом + os.replace (читатель не увидит недописанный файл)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def write_min_durations(links_output: str, segments: Sequence["SegmentSpec"]) -> str:
    """Пишет min_durations.json (пустой объект {}, если видео-сегментов нет). Возвращает путь."""
    path = min_durations_path(links_output)
    write_json_atomic(path, build_min_durations(segments))
    return path
_WM_BLOCKLIST_ENV = os.environ.get("SEARCH_WIKIMEDIA_BLOCKLIST")
WIKIMEDIA_BLOCKLIST_WORDS: tuple = (
    _DEFAULT_WM_BLOCKLIST if _WM_BLOCKLIST_ENV is None
    else tuple(w.strip().lower() for w in _WM_BLOCKLIST_ENV.split(",") if w.strip())
)

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


def _wm_word_in(word: str, text: str) -> bool:
    """Слово - по границам слов; многословная фраза - по вхождению."""
    if " " in word:
        return word in text
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None


def _wm_phrase_in(word: str, text: str) -> bool:
    """Слово или фраза по границам слов (для строгого блока: "id card" не найдётся в "paid card",
    "document" не найдётся в "documentary")."""
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None


def _wm_forms(w: str) -> set:
    forms = {w, w + "s"}
    if w.endswith("s"):
        forms.add(w[:-1])
    return forms


def _wm_blocked_impl(text: str, sc: str, blocklist, strict_words: frozenset) -> bool:
    for w in blocklist:
        w = (w or "").strip().lower()
        if not w:
            continue
        if w in strict_words:
            # строгое слово: формы единственного/множественного числа в тексте кандидата,
            # исключения по сцене нет
            if any(_wm_phrase_in(f, text) for f in _wm_forms(w)):
                return True
            continue
        if not _wm_word_in(w, text):
            continue
        if any(_wm_word_in(f, sc) for f in _wm_forms(w)):
            continue
        return True
    return False


def wikimedia_block_reason(
    candidate_text: str, scene: str, blocklist, strict: Optional[bool] = None,
) -> Optional[str]:
    """None - не блокируется; "base" - блокируется и при прежнем поведении; "strict" -
    блокируется только из-за строгого правила (слово из WIKIMEDIA_STRICT_BLOCK_WORDS есть
    в сцене, раньше это снимало блок)."""
    if not blocklist:
        return None
    if strict is None:
        strict = WIKIMEDIA_STRICT_BLOCK
    text = (candidate_text or "").lower()
    sc = (scene or "").lower()
    strict_words = frozenset(w.strip().lower() for w in WIKIMEDIA_STRICT_BLOCK_WORDS) if strict else frozenset()
    if not _wm_blocked_impl(text, sc, blocklist, strict_words):
        return None
    if strict and not _wm_blocked_impl(text, sc, blocklist, frozenset()):
        return "strict"
    return "base"


def wikimedia_blocked(candidate_text: str, scene: str, blocklist, strict: Optional[bool] = None) -> bool:
    """True, если в тексте кандидата есть слово блок-листа, которого нет в сцене сегмента
    (в сцене учитываются форма единственного/множественного числа). Слова из
    WIKIMEDIA_STRICT_BLOCK_WORDS блокируются независимо от сцены (если strict / env включён)."""
    return wikimedia_block_reason(candidate_text, scene, blocklist, strict) is not None


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
    blocked_total: int = 0
    blocked_strict_total: int = 0
    duration_rejected_total: int = 0  # видео отсеяно ранним фильтром длительности (до CLIP)
    duration_unknown_total: int = 0   # у видео нет/мусор в duration - пропущено без фильтра
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
            "%-18s %6d %7d %8d %9d %8d %9d %8d %8d %8d %7.3f %7.3f %8d %7d %8d" "%s",
            key, s.segments_attempted, s.raw_total, s.license_ok_total,
            s.keyword_ok_total, s.sent_to_clip_total, s.preview_missing_total,
            s.clip_scored_total, s.clip_passed_total, s.clip_accept_total,
            s.avg_score, s.best_score,
            s.accepted_total, s.rejected_foreign_total, s.best_effort_total,
            (f"  блок-лист={s.blocked_total} (из них строгих={s.blocked_strict_total})"
             if s.blocked_total else ""),
        )
    dur_parts = [
        f"{k}: отсеяно {site_stats[k].duration_rejected_total}, "
        f"длительность неизвестна: {site_stats[k].duration_unknown_total}"
        for k in sorted(site_stats)
        if site_stats[k].duration_rejected_total or site_stats[k].duration_unknown_total
    ]
    if dur_parts:
        logging.info("Фильтр видео по длительности (до CLIP): " + "; ".join(dur_parts))
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
    skip: bool = False  # select_coverage.py: True - сегмент не искать, сразу в missing
    min_duration: Optional[float] = None  # сек; нужен только видео-сегментам (type == "video")


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
    variant: Optional[str] = None  # narrow | medium | name | broad
    # Сырое значение duration из ответа поиска (только видео Pexels/Pixabay); None - поля нет.
    # Pexels: целые секунды; у Pixabay формат не подтверждён - разбор только через parse_duration_seconds.
    duration: Any = None
    # Тип медиа кандидата: "photo" | "video". Номера фото и видео у Pexels/Pixabay лежат в разных
    # пространствах, поэтому тип входит в ключ (см. cand_key). Заполняется в search_*.
    kind: str = ""


CAND_KINDS = ("photo", "video")


def kind_of_media_type(media_type: str) -> str:
    """Тип сегмента ("image"/"video") -> kind кандидата ("photo"/"video"). Неизвестное - ошибка."""
    if media_type == "image":
        return "photo"
    if media_type == "video":
        return "video"
    raise ValueError(f"Неизвестный тип медиа {media_type!r}: ожидается 'image' или 'video'")


def cand_key(cand: "Candidate") -> tuple:
    """Единый ключ кандидата (site, kind, cand_id): used_files, exclude_key, кэш векторов CLIP.
    Пустой или неизвестный kind - ошибка (молчаливых подстановок нет)."""
    if cand.kind not in CAND_KINDS:
        raise ValueError(
            f"У кандидата {cand.site}/{cand.cand_id} не задан kind (получено {cand.kind!r}): "
            f"ожидается один из {CAND_KINDS}"
        )
    return (cand.site, cand.kind, cand.cand_id)


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
    media_mode: int = MEDIA_MODE_DEFAULT  # 1 смешанный / 2 только видео / 3 только фото (см. parse_media_mode)
    # Переключение типа (только режим 1): номера сегментов, для которых пробовали другой тип;
    # номер -> ИТОГОВЫЙ тип (только если на другом типе кандидат найден); способ: abs / rank / best_effort.
    switch_attempted: set = field(default_factory=set)
    switch_final: dict = field(default_factory=dict)
    switch_how: Counter = field(default_factory=Counter)


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
    """Кэш нормализованных векторов превью по ключу (site, kind, cand_id). Один и тот же ключ
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
        # Серверный min_duration НЕ передаётся: у /videos/search его в документации нет (он описан
        # только для /videos/popular); не подтверждено живым запросом - не используем.
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
                    duration=v.get("duration"), kind="video",
                ))
        else:
            for p in data.get("photos", []):
                src = p.get("src") or {}
                preview = src.get("medium") or src.get("small") or src.get("original")
                result.append(Candidate(
                    site="pexels", cand_id=str(p["id"]), text=p.get("alt") or "",
                    license_ok=True, preview_url=preview, page_url=p.get("url"), kind="photo",
                ))
        return result

    return await cached_search(ctx, "pexels", media_type, query, _do)


async def search_pixabay(ctx: Context, query: str, media_type: str) -> list[Candidate]:
    async def _do() -> list[Candidate]:
        if "pixabay" in ctx.exhausted_sites:
            return []
        url = "https://pixabay.com/api/videos/" if media_type == "video" else "https://pixabay.com/api/"
        params = {"key": ctx.pixabay_api_key, "q": query, "per_page": 20}
        if PIXABAY_STRICT_TYPES:
            if media_type == "video":
                params["video_type"] = "film"
            else:
                params["image_type"] = "photo"
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
                duration=hit.get("duration") if media_type == "video" else None,
                kind=kind_of_media_type(media_type),
            ))
        return result

    return await cached_search(ctx, "pixabay", media_type, query, _do)


COMMONS_VIDEO_EXTENSIONS = frozenset({"webm", "ogv", "mpg", "mpeg"})
_commons_video_skip_logged = False


def site_skipped_in_video_only(site: str) -> bool:
    """Чистая функция: режим 2 (только видео) не обращается к сайту, если он не отдаёт видео из
    белого списка. Сейчас только wikimedia (Commons-видео webm/ogv/mpg/mpeg вне белого списка);
    считается от белого списка, т.е. сам снимется, если форматы Commons туда попадут.
    NASA (manifest-ресурсы mp4/mov) и LOC не пропускаются: их ответ не проверен вживую."""
    if site == "wikimedia":
        return not (COMMONS_VIDEO_EXTENSIONS & VIDEO_EXTENSIONS)
    return False


def video_only_skip_message(site: str) -> str:
    return f"режим 2: сайт {site} пропущен, видео не отдаёт"


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
                kind=kind_of_media_type(media_type),
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
                kind=kind_of_media_type(media_type),
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
                kind=kind_of_media_type(media_type),
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
            vec = await ctx.clip.cache.get_or_compute(cand_key(cand), compute)
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


async def try_claim_backup(
    ctx: Context, tail: list[Candidate], primary_site: str, seg_index: Optional[int] = None,
) -> Optional[str]:
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

    key = cand_key(backup_cand)
    async with ctx.used_files_lock:
        if key in ctx.used_files:
            return None
        ctx.used_files.add(key)
    url = await finalize_candidate(ctx, backup_cand)
    if url:
        _record_choice(ctx, backup_cand, seg_index, backup=True)
    return url


async def try_claim_pool(
    ctx: Context, pool: list[Candidate], seg_index: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    for i, cand in enumerate(pool):
        key = cand_key(cand)
        async with ctx.used_files_lock:
            if key in ctx.used_files:
                continue
            ctx.used_files.add(key)
        final_url = await finalize_candidate(ctx, cand)
        if final_url:
            if seg_index is not None:
                ctx.primary_cands[seg_index] = cand
            backup_url = await try_claim_backup(ctx, pool[i + 1:], cand.site, seg_index)
            return final_url, backup_url
    return None, None


ARCHIVE_SITES = ("wikimedia", "loc", "nasa")
STOCK_SITES = ("pexels", "pixabay")

# Максимум WARNING-строк про подстановку sites по режиму источников; остальное - одной итоговой строкой.
SOURCES_WARN_LIMIT = 10


def filter_sites_by_mode(sites, mode: Optional[int] = None) -> list:
    """Режим 1: только ARCHIVE_SITES; режим 3: только STOCK_SITES; режим 2: без изменений.
    Порядок и дубликаты сохраняются."""
    if mode is None:
        mode = SOURCES_MODE
    if mode == 1:
        return [x for x in sites if x in ARCHIVE_SITES]
    if mode == 3:
        return [x for x in sites if x in STOCK_SITES]
    return list(sites)


def fallback_sites(mode: int) -> list:
    return ["wikimedia", "loc"] if mode == 1 else ["pexels", "pixabay"]


def candidates_cap(
    site: str, override: Optional[int] = CANDIDATES_OVERRIDE,
    stock: int = CANDIDATES_STOCK, archive: int = CANDIDATES_ARCHIVE,
) -> int:
    """Потолок кандидатов с сайта под CLIP. override (SEARCH_CANDIDATES_PER_SITE) - для всех."""
    if override:
        return override
    return stock if site in STOCK_SITES else archive


def is_strong(c: Candidate, strong_own_sim: float = STRONG_OWN_SIM) -> bool:
    """Сильное принятие: по абсолютному порогу либо (ранг <= 2 и own_sim >= STRONG_OWN_SIM)."""
    return c.reject_reason == "abs" or (c.rank <= 2 and c.own_sim >= strong_own_sim)


@dataclass
class Variant:
    name: str  # "narrow" | "medium" | "name" | "broad"
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
    """Фильтр сущностей: только narrow/medium/name и только архивные сайты.
    (seg.is_entity и skip_keyword_filter проверяются в fetch_and_filter.)"""
    return variant_name in ("narrow", "medium", "name") and site in ARCHIVE_SITES


def _is_latin_text(t: str) -> bool:
    letters = [ch for ch in t if ch.isalpha()]
    return bool(letters) and all(unicodedata.name(ch, "").startswith("LATIN") for ch in letters)


def _entity_name_query(seg: SegmentSpec) -> Optional[str]:
    for kw in seg.entity_keywords or []:
        if kw and kw.strip() and _is_latin_text(kw.strip()):
            return kw.strip()
    return None


def build_cascade(seg: SegmentSpec, mode: Optional[int] = None) -> list[Variant]:
    """Чистая функция (без сети и ctx): упорядоченный список вариантов запроса.
    Режим источников 1 (только архивные): broad (он идёт только на pixabay/pexels) не строится."""
    if mode is None:
        mode = SOURCES_MODE
    seg_sites = [x for x in _pixabay_first(list(seg.sites)) if x in SITE_SEARCH_FUNCS]
    archive_first = bool(seg_sites) and seg_sites[0] in ARCHIVE_SITES

    if archive_first:
        order = ("narrow", "medium", "name", "broad") if seg.is_entity else ("narrow", "medium", "broad")
        narrow_medium_sites = [x for x in seg_sites if x in ARCHIVE_SITES]
    else:
        order = ("medium", "narrow", "broad")
        narrow_medium_sites = seg_sites
    queries = {
        "narrow": seg.query_narrow, "medium": seg.query_medium, "broad": seg.query_broad,
        "name": _entity_name_query(seg) if seg.is_entity else None,
    }
    broad_sites = _pixabay_first(list(STOCK_SITES))

    cascade: list[Variant] = []
    seen: list[tuple[str, set]] = []
    for name in order:
        q = queries[name]
        if not q or not q.strip():
            continue
        if name == "broad" and mode == 1:
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


def other_media_type(media_type: str) -> str:
    """Чистая функция: image <-> video. Иной тип - ValueError (молчаливых подстановок нет)."""
    if media_type == "image":
        return "video"
    if media_type == "video":
        return "image"
    raise ValueError(f"Неизвестный тип сегмента {media_type!r}: ожидается 'image' или 'video'")


def should_switch_type(media_mode: int, normal_found: bool) -> bool:
    """Чистая функция: пробовать ли другой тип. Только режим 1 и только если по родному типу
    не принят никто по нормальным условиям (abs/rank, до best-effort)."""
    return media_mode == 1 and not normal_found


def validate_switch_inputs(segments: Sequence["SegmentSpec"], media_mode: int) -> None:
    """Режим 1: любой сегмент, который может стать видео, обязан иметь min_duration (для проверки
    длительности и записи в min_durations.json). Нет - ValueError с номерами, подстановки нет."""
    if media_mode != 1:
        return
    bad = sorted(sg.index for sg in segments if not sg.skip and sg.min_duration is None)
    if bad:
        shown = ", ".join(map(str, bad[:20])) + (" ..." if len(bad) > 20 else "")
        raise ValueError(
            f"MEDIA_MODE=1: у сегментов нет min_duration в requests.json (номера: {shown}). "
            "Он нужен, чтобы сегмент мог стать видео при переключении типа; значение не подставляется."
        )


def effective_segment(seg: "SegmentSpec", switch_final: dict) -> "SegmentSpec":
    """Чистая функция: сегмент с ИТОГОВЫМ типом (после переключения), иначе тот же объект."""
    t = switch_final.get(seg.index)
    return seg if t is None or t == seg.type else dc_replace(seg, type=t)


def final_segments(segments: Sequence["SegmentSpec"], switch_final: dict) -> list:
    return [effective_segment(sg, switch_final) for sg in segments]


def alt_cascade(alt_seg: "SegmentSpec") -> list[Variant]:
    """Каскад для запасного типа (те же запросы). Для video сайты, не отдающие видео, убираются."""
    out: list[Variant] = []
    for v in build_cascade(alt_seg):
        if alt_seg.type == "video":
            sites = [x for x in v.sites if not site_skipped_in_video_only(x)]
            if not sites:
                continue
            v = dc_replace(v, sites=sites)
        out.append(v)
    return out


def summarize_type_switch(attempted: int, found: int, how: Counter) -> str:
    g = lambda k: int(how.get(k, 0))
    return (
        f"Переключение типа (режим 1): сегментов переключено {attempted}, из них найден кандидат {found} "
        f"(abs {g('abs')}, rank {g('rank')}, best-effort {g('best_effort')}), не найден {attempted - found}"
    )


def _stats_key(site: str, variant_name: str) -> str:
    return f"{site}_broad" if variant_name == "broad" else site


async def fetch_and_filter(
    ctx: Context, site: str, seg: SegmentSpec, query: str, variant_name: str,
    stats_key: Optional[str] = None,
) -> list[Candidate]:
    q = query
    # Режим 2 (только видео): сайт без видео в белом списке не опрашиваем. Строка в лог уже
    # записана в amain (одна на сайт), здесь - только возврат пустого результата.
    if ctx.media_mode == 2 and seg.type == "video" and site_skipped_in_video_only(site):
        return []
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

    # Ранний мягкий фильтр длительности (до превью и CLIP): только видео Pexels/Pixabay и только
    # если у сегмента есть min_duration. Кэш поиска общий для сегментов, поэтому фильтр здесь,
    # а не в search_*. Неизвестная длительность не отсеивает (строгая проверка - в download.py).
    if seg.type == "video" and seg.min_duration is not None and site in DURATION_FILTER_SITES:
        before_d = len(licensed)
        kept_d = []
        rejected_d = 0
        for c in licensed:
            verdict = duration_verdict(c.duration, seg.min_duration)
            if verdict == "reject":
                rejected_d += 1
                continue
            if verdict == "unknown":
                stats.duration_unknown_total += 1
            kept_d.append(c)
        stats.duration_rejected_total += rejected_d
        licensed = kept_d
        if rejected_d:
            logging.info(
                "Сегмент %s/%s [%s]: по длительности отсеяно %s из %s (min_duration=%.1f, порог: короче %.1f с).",
                seg.index, site, variant_name, rejected_d, before_d,
                seg.min_duration, seg.min_duration - DURATION_TOLERANCE_SECONDS,
            )
        if not licensed:
            logging.info(
                "Сегмент %s/%s [%s]: %s кандидатов прошли лицензию, но 0 после фильтра длительности.",
                seg.index, site, variant_name, before_d,
            )
            return []

    if site == "wikimedia" and WIKIMEDIA_BLOCKLIST_WORDS:
        before_bl = len(licensed)
        kept = []
        for c in licensed:
            reason = wikimedia_block_reason(c.text, seg.scene, WIKIMEDIA_BLOCKLIST_WORDS)
            if reason is None:
                kept.append(c)
            elif reason == "strict":
                stats.blocked_strict_total += 1
        licensed = kept
        stats.blocked_total += before_bl - len(licensed)
        if not licensed:
            logging.info(
                "Сегмент %s/%s [%s]: %s кандидатов прошли лицензию, но 0 после блок-листа типов контента.",
                seg.index, site, variant_name, before_bl,
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


LOG_META_MAX_LEN = 200


def pexels_slug(page_url: Optional[str]) -> str:
    """Слаг из page_url Pexels: последний сегмент пути без числового id в конце
    (.../video/some-title-1234567/ -> some-title). Если разобрать не удалось - сам page_url;
    пусто/None -> "". Только для лога, в отбор не участвует."""
    if not page_url:
        return ""
    try:
        segment = [x for x in urlparse(page_url).path.split("/") if x][-1]
    except (IndexError, ValueError):
        return page_url
    m = re.match(r"^(.+)-\d+$", segment)
    return m.group(1) if m else page_url


def candidate_log_meta(cand: Candidate, max_len: int = LOG_META_MAX_LEN) -> str:
    """Строка для лога: теги Pixabay (Candidate.text) или слаг Pexels; пусто, если данных нет.
    Только для лога: ни в фильтрации, ни в оценке, ни в выходных файлах не используется."""
    if cand.site == "pixabay":
        label, value = "теги", (cand.text or "").strip()
    elif cand.site == "pexels":
        label, value = "слаг", pexels_slug(cand.page_url).strip()
    else:
        return ""
    if not value:
        return ""
    if len(value) > max_len:
        value = value[:max_len] + "…"
    return f" {label}={value}"


def _record_choice(ctx: Context, cand: Candidate, seg_index: Optional[int] = None, backup: bool = False) -> None:
    key = _choice_key(cand)
    if backup:
        ctx.backup_reasons[key] += 1
        ctx.backup_own_sims.append(cand.own_sim)
        logging.info(
            "Сегмент %s: выбран резерв %s/%s [%s] own_sim=%.3f rank=%s причина=%s (%s)%s",
            seg_index, cand.site, cand.cand_id, cand.variant, cand.own_sim, cand.rank,
            cand.reject_reason,
            ("принят" if is_strong(cand) else "принят, слабый") if cand.accepted else "best-effort",
            candidate_log_meta(cand),
        )
        return
    ctx.choice_reasons[key] += 1
    ctx.choice_own_sims.append(cand.own_sim)
    logging.info(
        "Сегмент %s: выбран %s/%s [%s] own_sim=%.3f rank=%s причина=%s (%s)%s",
        seg_index, cand.site, cand.cand_id, cand.variant, cand.own_sim, cand.rank,
        cand.reject_reason,
        ("принят" if is_strong(cand) else "принят, слабый") if cand.accepted else "best-effort",
        candidate_log_meta(cand),
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
        top = licensed[:candidates_cap(site)]
        scored = await score_candidates(
            ctx, top, seg_index=seg.index, stats_key=key, variant_name=variant.name,
        )
        if not scored:
            continue
        pool.extend(scored)
        pool.sort(key=lambda c: c.own_sim, reverse=True)
        if any(c.accepted and is_strong(c) for c in scored):
            break
    return pool


async def _cascade_normal(
    ctx: Context, seg: SegmentSpec, cascade: list[Variant],
) -> tuple[Optional[str], Optional[str], list[Candidate]]:
    """Каскад с нормальными условиями (abs/rank) без best-effort. Возвращает (url, backup_url,
    rejected_pool): url None - никто не принят (или принятых не удалось заклеймить)."""
    rejected_pool: list[Candidate] = []
    weak_pool: list[Candidate] = []  # принятые только "впритык": каскад продолжается
    for variant in cascade:
        scored = await run_variant(ctx, seg, variant)
        accepted = select_accepted(seg, scored)
        if any(is_strong(c) for c in accepted):
            # Сильные первыми (основной выбор), остальные принятые - в хвост (под backup).
            ordered = [c for c in accepted if is_strong(c)] + [c for c in accepted if not is_strong(c)]
            url, backup_url = await try_claim_pool(ctx, ordered, seg.index)
            if url:
                chosen = ctx.primary_cands.get(seg.index)
                if chosen is not None:
                    _record_choice(ctx, chosen, seg.index)
                return url, backup_url, rejected_pool
        elif accepted:
            weak_pool.extend(dc_replace(c) for c in accepted)
        rejected_pool.extend(c for c in scored if not c.accepted)

    # Каскад кончился, сильных нет: лучший из всех слабых принятых по own_sim.
    if weak_pool:
        weak_pool.sort(key=lambda c: c.own_sim, reverse=True)
        url, backup_url = await try_claim_pool(ctx, weak_pool, seg.index)
        if url:
            chosen = ctx.primary_cands.get(seg.index)
            if chosen is not None:
                _record_choice(ctx, chosen, seg.index)
            return url, backup_url, rejected_pool
    return None, None, rejected_pool


async def process_segment_inner(ctx: Context, seg: SegmentSpec) -> tuple[Optional[str], Optional[str]]:
    # 1) родной тип, нормальные условия
    url, backup_url, rejected_native = await _cascade_normal(ctx, seg, build_cascade(seg))
    if url:
        return url, backup_url

    # 2) только режим 1: другой тип по тем же запросам, тоже нормальные условия
    alt: Optional[SegmentSpec] = None
    rejected_alt: list[Candidate] = []
    if should_switch_type(ctx.media_mode, False):
        alt = dc_replace(seg, type=other_media_type(seg.type))
        ctx.switch_attempted.add(seg.index)
        url, backup_url, rejected_alt = await _cascade_normal(ctx, alt, alt_cascade(alt))
        if url:
            chosen = ctx.primary_cands.get(seg.index)
            how = chosen.reject_reason if chosen is not None else "abs/rank"
            ctx.switch_final[seg.index] = alt.type
            ctx.switch_how[how] += 1
            logging.info("Сегмент %s: тип %s -> %s, найден по %s", seg.index, seg.type, alt.type, how)
            return url, backup_url

    # 3) Никто не принят ни по одному типу: best-effort. Берём лучшего среди оценённых по РОДНОМУ
    # типу; кандидатов другого типа - только если по родному не оценён ни один (иначе тип сегмента
    # не меняется из-за слабого кандидата).
    pool, pool_seg = rejected_native, seg
    if not pool and rejected_alt and alt is not None:
        pool, pool_seg = rejected_alt, alt
    if pool:
        url, backup_url = await try_claim_pool(ctx, best_effort_order(pool), seg.index)
        if url:
            chosen = ctx.primary_cands.get(seg.index)
            if chosen is not None:
                ctx.site_stats.setdefault(chosen.stats_key or chosen.site, SiteStats()).best_effort_total += 1
                _record_choice(ctx, chosen, seg.index)
            if pool_seg is not seg:
                ctx.switch_final[seg.index] = pool_seg.type
                ctx.switch_how["best_effort"] += 1
                logging.info(
                    "Сегмент %s: тип %s -> %s, найден по best-effort (по родному типу кандидатов нет)",
                    seg.index, seg.type, pool_seg.type,
                )
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
            if cand_key(c) not in ctx.used_files
            and cand_key(c) != exclude_key
        ]
    top = fresh[:candidates_cap(site)]
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
        key = cand_key(cand)
        async with ctx.used_files_lock:
            if key in ctx.used_files:
                continue
            ctx.used_files.add(key)
        url = await finalize_candidate(ctx, cand)
        if url:
            return url, cand
    return None, None


def backup_candidate_sites(seg_sites, primary_site: str, mode: Optional[int] = None) -> list:
    """Сайты для backup: seg.sites + SEARCH_BACKUP_EXTRA_SITES (+ pexels/pixabay, если primary - loc).
    Режим источников фильтрует результат целиком (режим 2 - без изменений)."""
    candidate_sites = list(seg_sites) + BACKUP_EXTRA_SITES
    # Если primary был loc, обязательно добавляем pexels и pixabay в список резерва
    if primary_site == "loc":
        for stock_site in ("pexels", "pixabay"):
            if stock_site not in candidate_sites:
                candidate_sites.append(stock_site)
    return filter_sites_by_mode(candidate_sites, mode)


async def find_backup(ctx: Context, seg: SegmentSpec) -> tuple[Optional[str], Optional[str]]:
    """B1 (другие сайты) -> B2 (тот же сайт). Возвращает (url, 'other'|'same') или (None, None)."""
    primary = ctx.primary_cands.get(seg.index)
    if primary is None:
        return None, None
    pkey = cand_key(primary)

    cascade = build_cascade(seg)

    candidate_sites = backup_candidate_sites(seg.sites, primary.site)

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
                        _record_choice(ctx, _c, seg.index, backup=True)
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
                        _record_choice(ctx, _c, seg.index, backup=True)
                    return url, "same"
                rejected.extend(c for c in pool if not c.accepted)

    # Принятых бэкапов нет: бэкап лучше иметь, чем нет - берём лучшего по own_sim.
    if rejected:
        url, cand = await _claim_first(ctx, best_effort_order(rejected))
        if url and cand is not None:
            _record_choice(ctx, cand, seg.index, backup=True)
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
                # backup того же типа, что и основной (итоговый тип после переключения)
                url, src = await find_backup(ctx, effective_segment(seg, ctx.switch_final))
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


def split_skipped(specs: list[SegmentSpec]) -> tuple[list[SegmentSpec], list[int]]:
    """(сегменты для поиска, отсортированные номера сегментов со skip == True)."""
    return [sg for sg in specs if not sg.skip], sorted(sg.index for sg in specs if sg.skip)


def load_requests(path: str) -> list[SegmentSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("requests.json пуст или имеет неверную структуру (ожидался объект-словарь)")

    def bad(k, field, what):
        return ValueError(f"Сегмент {k!r} в requests.json: поле {field!r} {what}")

    specs: list[SegmentSpec] = []
    sources_warned = 0
    sources_suppressed = 0
    video_without_md: list[int] = []
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
        skip = v.get("skip")
        if skip is None:
            skip = False  # поля нет (старый requests.json) или null - искать как раньше
        elif not isinstance(skip, bool):
            raise bad(k, "skip", "должно быть true или false")
        raw_md = v.get("min_duration")
        min_duration: Optional[float] = None
        if raw_md is not None:
            if (isinstance(raw_md, bool) or not isinstance(raw_md, (int, float))
                    or not math.isfinite(raw_md) or raw_md < 0):
                raise bad(k, "min_duration", "должно быть неотрицательным числом (секунды) или null")
            min_duration = float(raw_md)
        if seg_type == "video" and not skip and min_duration is None:
            video_without_md.append(idx)
        norm_sites = [x.strip().lower() for x in sites]
        unknown = [x for x in norm_sites if x not in SITE_SEARCH_FUNCS]
        if unknown:
            logging.warning(
                "Сегмент %s: неизвестные сайты в sites %s - они будут проигнорированы.", k, unknown,
            )
        if SOURCES_MODE != 2:
            filtered = filter_sites_by_mode(norm_sites)
            if not filtered:
                filtered = fallback_sites(SOURCES_MODE)
                if sources_warned < SOURCES_WARN_LIMIT:
                    sources_warned += 1
                    logging.warning(
                        "Сегмент %s: после фильтра режима источников %s в sites %s ничего не осталось - "
                        "подставлено %s.", k, SOURCES_MODE, norm_sites, filtered,
                    )
                else:
                    sources_suppressed += 1
            norm_sites = filtered
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
                skip=skip,
                min_duration=min_duration,
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Сегмент {k!r} в requests.json имеет некорректную структуру: {e}") from e

    if video_without_md:
        shown = ", ".join(str(i) for i in sorted(video_without_md)[:20])
        more = f" и ещё {len(video_without_md) - 20}" if len(video_without_md) > 20 else ""
        raise ValueError(
            f"У видео-сегментов нет поля min_duration в requests.json (номера: {shown}{more}). "
            "Перегенерируйте requests.json; значение по умолчанию не подставляется."
        )
    if sources_suppressed:
        logging.warning(
            "Режим источников %s: ещё у %s сегментов sites после фильтра был пуст "
            "(подставлены стандартные; подробные строки подавлены, лимит %s).",
            SOURCES_MODE, sources_suppressed, SOURCES_WARN_LIMIT,
        )
    specs.sort(key=lambda s: s.index)
    return specs


async def amain(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.input):
        logging.error("Входной файл не найден: %s", args.input)
        return 1

    try:
        media_mode = parse_media_mode(os.environ.get("MEDIA_MODE"))  # единственное место чтения env
    except ValueError as e:
        logging.error("%s", e)
        return 1
    logging.info("Режим медиа: %s (%s; MEDIA_MODE).", media_mode, MEDIA_MODE_NAMES[media_mode])
    if media_mode == 2:
        for _site in SITE_SEARCH_FUNCS:
            if site_skipped_in_video_only(_site):
                logging.info(video_only_skip_message(_site))

    if not _SOURCES_MODE_VALID:
        logging.warning(
            "SEARCH_SOURCES_MODE=%r не из {1,2,3} - игнорируется, берётся %s.",
            _SOURCES_MODE_RAW, SOURCES_MODE_DEFAULT,
        )
    logging.info(
        "Режим источников: %s (%s; SEARCH_SOURCES_MODE).", SOURCES_MODE,
        {1: "только архивные: wikimedia/loc/nasa, broad не выполняется",
         2: "микс, без ограничений",
         3: "только стоки: pexels/pixabay"}[SOURCES_MODE],
    )
    try:
        all_segments = load_requests(args.input)
        validate_switch_inputs(all_segments, media_mode)
    except (ValueError, json.JSONDecodeError) as e:
        logging.error("Ошибка чтения %s: %s", args.input, e)
        return 1

    # skip == True: не ищем (ни сайты, ни CLIP, ни каскад, ни backup), в конце идут в missing.
    # Дальше `segments` - только сегменты для поиска, поэтому encode_scenes, rel_top_n,
    # прогресс run_search и backup-проход считаются от них, а не от полного списка.
    segments, skipped = split_skipped(all_segments)
    logging.info("Загружено сегментов: %s", len(all_segments))
    if skipped:
        logging.info(
            "Пропущено по skip (не ищутся, попадут в %s как не найденные): %s; к поиску: %s.",
            args.missing_output, len(skipped), len(segments),
        )
    if not segments:
        logging.warning("Все сегменты помечены skip - поиск не выполняется, CLIP не загружается.")
        for path, lines in ((args.links_output, []), (args.backup_links_output, []),
                            (args.missing_output, [f"{i}\n" for i in skipped]),
                            (args.backup_missing_output, [])):
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        write_min_durations(args.links_output, all_segments)  # все skip -> {}
        logging.info(
            "Готово: найдено 0 из 0 сегментов, пропущено по skip %s (все записаны в %s).",
            len(skipped), args.missing_output,
        )
        return 0
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

    logging.info(
        "Параметры поиска: STRONG_OWN_SIM=%.3f (SEARCH_STRONG_OWN_SIM); потолки кандидатов: "
        "сток=%s, архивы=%s, общий override=%s (SEARCH_CANDIDATES_STOCK / SEARCH_CANDIDATES_ARCHIVE / "
        "SEARCH_CANDIDATES_PER_SITE); блок-лист Wikimedia: %s слов (SEARCH_WIKIMEDIA_BLOCKLIST), строгий блок без исключения по "
        "сцене: %s (SEARCH_WIKIMEDIA_STRICT_BLOCK); типы контента Pixabay (video_type=film / "
        "image_type=photo): %s (SEARCH_PIXABAY_STRICT_TYPES).",
        STRONG_OWN_SIM, CANDIDATES_STOCK, CANDIDATES_ARCHIVE,
        CANDIDATES_OVERRIDE if CANDIDATES_OVERRIDE else "нет", len(WIKIMEDIA_BLOCKLIST_WORDS),
        "вкл" if WIKIMEDIA_STRICT_BLOCK else "выкл",
        "вкл" if PIXABAY_STRICT_TYPES else "выкл",
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
            media_mode=media_mode,
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

        # Пропущенные по skip - в тот же список missing (формат/сортировка при записи общие).
        # backup-проход ниже идёт по results, пропущенных там нет.
        missing.extend(skipped)

        backup_missing = await run_backup_pass(ctx, segments, results, backups)

        logging.info(summarize_choices(ctx.choice_reasons, ctx.choice_own_sims, "primary"))
        logging.info(summarize_choices(ctx.backup_reasons, ctx.backup_own_sims, "backup"))
        if media_mode == 1:
            logging.info(summarize_type_switch(
                len(ctx.switch_attempted), len(ctx.switch_final), ctx.switch_how,
            ))

        log_site_stats_summary(ctx.site_stats)

    with open(args.links_output, "w", encoding="utf-8") as f:
        for idx in sorted(results):
            f.write(f"{idx}: {results[idx]}\n")

    with open(args.backup_links_output, "w", encoding="utf-8") as f:
        for idx in sorted(backups):
            f.write(f"{idx}: {backups[idx]}\n")

    # по ИТОГОВОМУ типу (после переключения и backup-прохода)
    final_segs = final_segments(all_segments, ctx.switch_final)
    md_path = write_min_durations(args.links_output, final_segs)
    logging.info("Записан %s (видео-сегментов: %s).", md_path, len(build_min_durations(final_segs)))

    with open(args.missing_output, "w", encoding="utf-8") as f:
        for idx in sorted(missing):
            f.write(f"{idx}\n")

    with open(args.backup_missing_output, "w", encoding="utf-8") as f:
        for idx in backup_missing:
            f.write(f"{idx}\n")

    reject_line = ctx.format_rejects.summary_line()
    if reject_line:
        logging.info(reject_line)

    # Статистика считается от сегментов, которые реально искались (len(segments)); пропущенные
    # по skip в неё не входят и показаны отдельной строкой ниже (в missing.txt они есть).
    logging.info(
        "Готово: найдено %s из %s сегментов (из них с backup - %s), не найдено %s, "
        "backup_missing %s. Результаты: %s, backup: %s, пропуски: %s, без backup: %s",
        len(results), len(segments), len(backups), len(missing) - len(skipped), len(backup_missing),
        args.links_output, args.backup_links_output, args.missing_output,
        args.backup_missing_output,
    )
    if skipped:
        logging.info(
            "Пропущено по skip: %s (не искались; не входят в числа выше; всего сегментов %s, "
            "строк в %s: %s).",
            len(skipped), len(all_segments), args.missing_output, len(missing),
        )
    return 0


class _CaptureWarnings(logging.Handler):
    """Для self-test: собирает тексты WARNING-записей."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())

    def __enter__(self):
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, *a):
        logging.getLogger().removeHandler(self)
        return False


def _selftest() -> int:
    import tempfile

    # self-test не зависит от SEARCH_SOURCES_MODE / SEARCH_WIKIMEDIA_STRICT_BLOCK в окружении
    globals()["SOURCES_MODE"] = 2
    globals()["WIKIMEDIA_STRICT_BLOCK"] = True
    _selftest_pixabay_strict_env = PIXABAY_STRICT_TYPES  # реальное значение из окружения (для проверки env-режима)

    def mk(sites, n="wiki narrow", m="wiki medium", b="city street", **kw):
        return SegmentSpec(index=1, scene="s", sites=sites, query_narrow=n, query_medium=m,
                           query_broad=b, type="image", is_entity=True,
                           entity_keywords=kw.get("ek", []))

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
                  "query_broad": "", "type": "video", "is_entity": False, "entity_keywords": [],
                  "min_duration": 5.0}}
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
    # skip: нет поля / null -> False; true/false читаются; не bool -> ошибка
    assert specs[0].skip is False
    assert load({"1": dict(good["1"], skip=None)})[0].skip is False
    assert load({"1": dict(good["1"], skip=True)})[0].skip is True
    assert load({"1": dict(good["1"], skip=False)})[0].skip is False
    for bad_skip in ("true", 1, 0):
        try:
            load({"1": dict(good["1"], skip=bad_skip)})
        except ValueError as e:
            assert "skip" in str(e) and "Сегмент" in str(e), e
        else:
            raise AssertionError("ожидалась ValueError для skip=%r" % (bad_skip,))
    # skip-сегменты не идут в поиск и оказываются в missing; без skip - как раньше
    _sp = load({"1": good["1"], "2": dict(good["1"], skip=True), "3": dict(good["1"], skip=False),
                "10": dict(good["1"], skip=True)})
    _act, _skp = split_skipped(_sp)
    assert [x.index for x in _act] == [1, 3] and _skp == [2, 10], (_act, _skp)
    _act, _skp = split_skipped(load({"1": good["1"], "2": good["1"]}))
    assert [x.index for x in _act] == [1, 2] and _skp == []

    _searched: list[int] = []

    async def _spy_process(ctx, seg):
        _searched.append(seg.index)
        return seg.index, (None if seg.index == 3 else f"u{seg.index}"), None

    _old_ps = globals()["process_segment"]
    globals()["process_segment"] = _spy_process
    try:
        _act, _skp = split_skipped(_sp)
        _res_, _bk_, _miss_ = asyncio.run(run_search(None, _act))
    finally:
        globals()["process_segment"] = _old_ps
    assert _searched == [1, 3] and sorted(_res_) == [1] and _miss_ == [3], (_searched, _res_, _miss_)
    _miss_.extend(_skp)  # ровно так amain собирает список для missing.txt ("номер" на строку)
    assert sorted(_miss_) == [2, 3, 10]
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
    # is_strong
    sc = lambda r, x, why: Candidate("p", "1", "", True, None, None, own_sim=x, similarity=x,
                                     rank=r, accepted=True, reject_reason=why)
    assert is_strong(sc(5, 0.30, "abs")) is True
    assert is_strong(sc(1, 0.25, "rank")) is True
    assert is_strong(sc(4, 0.214, "rank")) is False
    assert is_strong(sc(2, 0.23, "rank")) is False
    # вариант name
    def mk2(sites, ek, ent=True, n="Mehmed VI Constantinople 1920", m="Mehmed VI sultan", b="ottoman palace"):
        return SegmentSpec(index=1, scene="s", sites=sites, query_narrow=n, query_medium=m,
                           query_broad=b, type="image", is_entity=ent, entity_keywords=ek)
    c = view(mk2(["wikimedia", "loc"], ["Mehmed VI", "Мехмед VI"]))
    assert [n for n, _ in c] == ["narrow", "medium", "name", "broad"], c
    assert c[2] == ("name", ["wikimedia", "loc"]), c
    assert [v.query for v in build_cascade(mk2(["wikimedia", "loc"], ["Mehmed VI", "Мехмед VI"]))][2] == "Mehmed VI"
    assert [v.query for v in build_cascade(mk2(["wikimedia"], ["Мехмед", "Treaty of Sevres"]))][2] == "Treaty of Sevres"
    assert all(n != "name" for n, _ in view(mk2(["wikimedia"], ["Мехмед VI"])))
    assert all(n != "name" for n, _ in view(mk2(["wikimedia"], [])))
    assert all(n != "name" for n, _ in view(mk2(["wikimedia"], ["Mehmed VI"], ent=False)))
    assert all(n != "name" for n, _ in view(mk2(["pexels", "wikimedia"], ["Mehmed VI"])))
    assert all(n != "name" for n, _ in view(mk2(["wikimedia"], ["Mehmed VI"], m="mehmed  vi")))
    assert entity_filter_applies(mk2(["wikimedia"], ["x"]), "name", "wikimedia") is True
    assert entity_filter_applies(mk2(["wikimedia"], ["x"]), "name", "pexels") is False
    # candidates_cap (без os.environ)
    for st in ("pexels", "pixabay"):
        assert candidates_cap(st, None, 15, 8) == 15
    for st in ("wikimedia", "loc", "nasa"):
        assert candidates_cap(st, None, 15, 8) == 8
    assert candidates_cap("pexels", 5, 15, 8) == 5 and candidates_cap("loc", 5, 15, 8) == 5
    # wikimedia_blocked
    bl = WIKIMEDIA_BLOCKLIST_WORDS or _DEFAULT_WM_BLOCKLIST
    assert wikimedia_blocked("Stamps of Russia 2013", "Sultan leaves the palace", bl) is True
    assert wikimedia_blocked("Stamps of Russia 2013", "a postage stamp is shown", bl) is False
    assert wikimedia_blocked("Stamps of Russia 2013", "Sultan", ()) is False
    assert wikimedia_blocked("Mapleton street 1920", "street", bl) is False
    assert wikimedia_blocked("Old book cover scan", "a book", bl) is True

    # --- блок-лист: строгие слова ---
    sbl = _DEFAULT_WM_BLOCKLIST
    assert wikimedia_blocked("Old map of Constantinople", "a map of the city", sbl, strict=True) is True
    assert wikimedia_blocked("Old maps of Constantinople", "a map of the city", sbl, strict=True) is True
    assert wikimedia_blocked("Old map of Constantinople", "a map of the city", sbl, strict=False) is False
    assert wikimedia_block_reason("Old map of Constantinople", "a map of the city", sbl, True) == "strict"
    assert wikimedia_block_reason("Old map of Constantinople", "a map of the city", sbl, False) is None
    assert wikimedia_block_reason("Old map of Constantinople", "palace", sbl, True) == "base"
    assert wikimedia_block_reason("Palace garden", "palace", sbl, True) is None
    assert wikimedia_blocked("Mapleton street 1920", "a map of the city", sbl, strict=True) is False
    assert wikimedia_blocked("Documentary about Rome", "document", sbl, strict=True) is False
    assert wikimedia_blocked("Scanned documents 1920", "palace", sbl, strict=True) is True
    assert wikimedia_blocked("Soviet passport 1974", "passport office", sbl, strict=True) is True
    assert wikimedia_blocked("ID card of a clerk", "office", sbl, strict=True) is True
    assert wikimedia_blocked("Paid cardinal portrait", "office", sbl, strict=True) is False
    assert wikimedia_blocked("Coat of arms of Rome", "coat of arms", sbl, strict=True) is True
    # обычное слово блок-листа: исключение по сцене работает как раньше
    assert wikimedia_blocked("Stamps of Russia 2013", "a postage stamp is shown", sbl, strict=True) is False
    assert wikimedia_blocked("Stamps of Russia 2013", "Sultan", sbl, strict=True) is True
    # выключение строгого блока (SEARCH_WIKIMEDIA_STRICT_BLOCK=0) возвращает прежнее поведение
    _g = globals()
    _old_strict = _g["WIKIMEDIA_STRICT_BLOCK"]
    try:
        _g["WIKIMEDIA_STRICT_BLOCK"] = True
        assert wikimedia_blocked("Old map", "a map", sbl) is True
        _g["WIKIMEDIA_STRICT_BLOCK"] = False
        assert wikimedia_blocked("Old map", "a map", sbl) is False
        assert wikimedia_blocked("Old map", "palace", sbl) is True
    finally:
        _g["WIKIMEDIA_STRICT_BLOCK"] = _old_strict
    # слова строгого списка есть в дефолтном блок-листе
    for w in WIKIMEDIA_STRICT_BLOCK_WORDS:
        assert w.lower() in _DEFAULT_WM_BLOCKLIST, w

    # --- режим источников ---
    assert parse_sources_mode(None) == (2, True)
    assert parse_sources_mode("") == (2, True)
    assert parse_sources_mode("1") == (1, True)
    assert parse_sources_mode(" 3 ") == (3, True)
    assert parse_sources_mode("4") == (2, False)
    assert parse_sources_mode("abc") == (2, False)
    mixed = ["pexels", "wikimedia", "pixabay", "loc", "nasa"]
    assert filter_sites_by_mode(mixed, 1) == ["wikimedia", "loc", "nasa"]
    assert filter_sites_by_mode(mixed, 3) == ["pexels", "pixabay"]
    assert filter_sites_by_mode(mixed, 2) == mixed
    assert filter_sites_by_mode(["pexels"], 1) == []
    # broad в режиме 1 не выполняется, в 2 и 3 - как раньше
    seg_a = mk(["wikimedia", "pexels"])
    assert [n for n, _ in [(v.name, v.sites) for v in build_cascade(seg_a, 1)]] == ["narrow", "medium"]
    assert [n for n, _ in [(v.name, v.sites) for v in build_cascade(seg_a, 2)]] == ["narrow", "medium", "broad"]
    assert [n for n, _ in [(v.name, v.sites) for v in build_cascade(seg_a, 3)]] == ["narrow", "medium", "broad"]
    seg_s = mk(["pexels", "pixabay"])
    assert [v.name for v in build_cascade(seg_s, 3)] == ["medium", "narrow", "broad"]

    # load_requests: режимы 1/3, пустой sites после фильтра, лимит warning'ов
    def _write_req(sites_list):
        data = {}
        for i, st in enumerate(sites_list, 1):
            data[str(i)] = {"scene": "s", "sites": st, "query_narrow": "n", "query_medium": "m",
                            "query_broad": "b", "type": "image", "is_entity": False,
                            "entity_keywords": []}
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return path

    _old_mode = _g["SOURCES_MODE"]
    try:
        path = _write_req([["pexels", "wikimedia", "loc"], ["pexels", "pixabay"], ["wikimedia"]])
        _g["SOURCES_MODE"] = 1
        r = load_requests(path)
        assert [x.sites for x in r] == [["wikimedia", "loc"], ["wikimedia", "loc"], ["wikimedia"]], r
        _g["SOURCES_MODE"] = 3
        r = load_requests(path)
        assert [x.sites for x in r] == [["pexels"], ["pexels", "pixabay"], ["pexels", "pixabay"]], r
        _g["SOURCES_MODE"] = 2
        r = load_requests(path)
        assert [x.sites for x in r] == [["pexels", "wikimedia", "loc"], ["pexels", "pixabay"], ["wikimedia"]]
        os.remove(path)
        # пустой после фильтра: WARNING с номером сегмента, лимит строк
        path = _write_req([["pexels"]] * (SOURCES_WARN_LIMIT + 3))
        _g["SOURCES_MODE"] = 1
        with _CaptureWarnings() as cap:
            r = load_requests(path)
        assert all(x.sites == ["wikimedia", "loc"] for x in r)
        per_seg = [m for m in cap.msgs if m.startswith("Сегмент ")]
        assert len(per_seg) == SOURCES_WARN_LIMIT and "Сегмент 1:" in per_seg[0], cap.msgs
        assert any("подавлены" in m for m in cap.msgs), cap.msgs
        os.remove(path)
    finally:
        _g["SOURCES_MODE"] = _old_mode


    # backup-сайты по режимам
    _old_extra = list(BACKUP_EXTRA_SITES)
    try:
        BACKUP_EXTRA_SITES[:] = ["nasa", "pixabay"]
        assert backup_candidate_sites(["wikimedia", "pexels"], "wikimedia", 2) == \
            ["wikimedia", "pexels", "nasa", "pixabay"]
        assert backup_candidate_sites(["wikimedia", "pexels"], "wikimedia", 1) == ["wikimedia", "nasa"]
        assert backup_candidate_sites(["wikimedia", "pexels"], "wikimedia", 3) == ["pexels", "pixabay"]
        assert backup_candidate_sites(["loc"], "loc", 1) == ["loc", "nasa"]
        assert backup_candidate_sites(["loc"], "loc", 2) == ["loc", "nasa", "pixabay", "pexels"]
        assert backup_candidate_sites(["loc"], "loc", 3) == ["pixabay", "pexels"]
    finally:
        BACKUP_EXTRA_SITES[:] = _old_extra

    # --- Pixabay: типы контента в параметрах запроса ---
    def _pixabay_params(media_type, strict):
        captured = []

        async def _fake_http(ctx_, site, url, headers=None, params=None, treat_429_as_exhaustion=False):
            captured.append((url, dict(params or {})))
            return {"hits": []}

        class _Ctx:
            exhausted_sites: set = set()
            pixabay_api_key = "K"
            search_cache: dict = {}
            search_cache_lock = asyncio.Lock()

        _old_http, _old_flag = _g["http_get_json"], _g["PIXABAY_STRICT_TYPES"]
        _g["http_get_json"], _g["PIXABAY_STRICT_TYPES"] = _fake_http, strict
        try:
            asyncio.run(search_pixabay(_Ctx(), "q " + media_type + str(strict), media_type))
        finally:
            _g["http_get_json"], _g["PIXABAY_STRICT_TYPES"] = _old_http, _old_flag
        assert len(captured) == 1, captured
        return captured[0][1]

    pv = _pixabay_params("video", True)
    assert pv.get("video_type") == "film" and "image_type" not in pv, pv
    assert pv["key"] == "K" and pv["per_page"] == 20 and pv["q"].startswith("q video"), pv
    pi = _pixabay_params("image", True)
    assert pi.get("image_type") == "photo" and "video_type" not in pi, pi
    assert pi["key"] == "K" and pi["per_page"] == 20, pi
    for _mt in ("video", "image"):
        p0 = _pixabay_params(_mt, False)
        assert "video_type" not in p0 and "image_type" not in p0, p0
        assert set(p0) == {"key", "q", "per_page"}, p0
    # значение константы из окружения согласовано с разбором SEARCH_PIXABAY_STRICT_TYPES
    _env_v = os.environ.get("SEARCH_PIXABAY_STRICT_TYPES", "1").strip().lower()
    assert _selftest_pixabay_strict_env == (_env_v not in ("0", "false", "no", "off"))

    # --- слаг Pexels ---
    assert pexels_slug("https://www.pexels.com/video/some-title-1234567/") == "some-title"
    assert pexels_slug("https://www.pexels.com/photo/a-man-on-a-beach-98765") == "a-man-on-a-beach"
    assert pexels_slug("https://www.pexels.com/video/1234567/") == "https://www.pexels.com/video/1234567/"
    assert pexels_slug("https://www.pexels.com/video/no-id-here/") == "https://www.pexels.com/video/no-id-here/"
    assert pexels_slug("") == "" and pexels_slug(None) == ""

    # --- лог выбранного кадра: теги/слаг только в логе, выходные данные не меняются ---
    class _CaptureInfo(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.INFO)
            self.msgs: list[str] = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    class _LogCtx:
        def __init__(self):
            self.choice_reasons, self.choice_own_sims = Counter(), []
            self.backup_reasons, self.backup_own_sims = Counter(), []

    def _logged(cand, backup=False):
        h, root = _CaptureInfo(), logging.getLogger()
        _lvl = root.level
        root.addHandler(h)
        root.setLevel(logging.INFO)
        try:
            _record_choice(_LogCtx(), cand, 7, backup=backup)
        finally:
            root.removeHandler(h)
            root.setLevel(_lvl)
        assert len(h.msgs) == 1, h.msgs
        return h.msgs[0]

    c_pb = Candidate(site="pixabay", cand_id="42", text="sea, wave, animation", license_ok=True,
                     preview_url=None, page_url="https://pixabay.com/videos/x-42/",
                     own_sim=0.3, rank=1, accepted=True, reject_reason="abs", variant="broad", kind="video")
    c_px = Candidate(site="pexels", cand_id="1234567", text="", license_ok=True, preview_url=None,
                     page_url="https://www.pexels.com/video/ocean-waves-1234567/",
                     own_sim=0.3, rank=1, accepted=True, reject_reason="abs", variant="broad", kind="video")
    c_wm = Candidate(site="wikimedia", cand_id="File:A.jpg", text="x", license_ok=True, preview_url=None,
                     page_url="https://commons.wikimedia.org/wiki/File:A.jpg",
                     own_sim=0.3, rank=1, accepted=True, reject_reason="abs", variant="narrow")
    for _bk in (False, True):
        m = _logged(c_pb, _bk)
        assert "pixabay/42" in m and "теги=sea, wave, animation" in m, m
        m = _logged(c_px, _bk)
        assert "pexels/1234567" in m and "слаг=ocean-waves" in m and "page_url" not in m, m
        m = _logged(c_wm, _bk)
        assert "теги=" not in m and "слаг=" not in m, m
    assert ("резерв" in _logged(c_pb, True)) and ("резерв" not in _logged(c_pb, False))
    # пустые теги не печатаются; длинные обрезаются
    c_empty = Candidate(site="pixabay", cand_id="1", text="  ", license_ok=True, preview_url=None, page_url=None)
    assert candidate_log_meta(c_empty) == ""
    assert candidate_log_meta(Candidate(site="pexels", cand_id="1", text="", license_ok=True,
                                        preview_url=None, page_url=None)) == ""
    c_long = Candidate(site="pixabay", cand_id="1", text="t" * 500, license_ok=True, preview_url=None, page_url=None)
    assert len(candidate_log_meta(c_long)) <= len(" теги=") + LOG_META_MAX_LEN + 1
    # логирование не меняет кандидата и не влияет на URL, уходящие в links.txt / backup_links.txt
    _before = dc_replace(c_pb)
    _logged(c_pb)
    assert c_pb == _before
    import types as _types

    async def _fake_finalize(ctx_, cand_):
        return f"https://cdn.example/{cand_.site}/{cand_.cand_id}.mp4"

    _old_fin = _g["finalize_candidate"]
    _g["finalize_candidate"] = _fake_finalize
    try:
        _uctx = _types.SimpleNamespace(
            used_files=set(), used_files_lock=asyncio.Lock(), primary_cands={},
            choice_reasons=Counter(), choice_own_sims=[], backup_reasons=Counter(), backup_own_sims=[],
        )
        _res = asyncio.run(try_claim_pool(_uctx, [c_pb, c_px], 7))
    finally:
        _g["finalize_candidate"] = _old_fin
    assert _res == ("https://cdn.example/pixabay/42.mp4", "https://cdn.example/pexels/1234567.mp4"), _res
    # формат строк выходных файлов остаётся прежним: "номер: URL" и "номер"
    assert f"{7}: {_res[0]}\n" == "7: https://cdn.example/pixabay/42.mp4\n"
    assert f"{7}: {_res[1]}\n" == "7: https://cdn.example/pexels/1234567.mp4\n"
    assert "теги" not in _res[0] and "слаг" not in _res[1]

    # ---- ЧАСТЬ А: MEDIA_MODE, фильтр длительности, min_durations.json ----
    # parse_media_mode
    assert parse_media_mode(None) == 1 and parse_media_mode("") == 1 and parse_media_mode("  ") == 1
    assert [parse_media_mode(x) for x in ("1", "2", "3", " 2 ")] == [1, 2, 3, 2]
    for bad_mode in ("0", "4", "2.0", "video", "01", "-1", "1,2"):
        try:
            parse_media_mode(bad_mode)
        except ValueError as e:
            assert "MEDIA_MODE" in str(e) and "только видео" in str(e), e
        else:
            raise AssertionError("ожидалась ValueError для MEDIA_MODE=%r" % (bad_mode,))

    # parse_duration_seconds / duration_verdict: граница min-1, нет поля, мусор
    assert parse_duration_seconds(12) == 12.0 and parse_duration_seconds(7.5) == 7.5
    for junk in (None, "", "12", "abc", 0, 0.0, -3, -0.5, float("nan"), float("inf"), True, False, [], {}):
        assert parse_duration_seconds(junk) is None, junk
        assert duration_verdict(junk, 10.0) == "unknown", junk
    assert duration_verdict(8.9, 10.0) == "reject"      # 8.9 < 9.0
    assert duration_verdict(9, 10.0) == "keep"          # ровно min-1: не отсеиваем
    assert duration_verdict(9.0, 10.0) == "keep"
    assert duration_verdict(9.01, 10.0) == "keep"
    assert duration_verdict(10, 10.0) == "keep" and duration_verdict(30, 10.0) == "keep"
    assert duration_verdict(4, 5.3) == "reject" and duration_verdict(5, 5.3) == "keep"
    assert duration_verdict(1, 1.0) == "keep" and duration_verdict(1, 0.0) == "keep"

    # build_min_durations / запись
    def _sg(i, t, md, skip=False):
        return SegmentSpec(index=i, scene="s", sites=["pexels"], query_narrow="n", query_medium="m",
                           query_broad=None, type=t, is_entity=False, entity_keywords=[],
                           skip=skip, min_duration=md)

    _segs = [_sg(10, "video", 7.25), _sg(2, "video", 5.0), _sg(3, "image", None),
             _sg(4, "video", 9.0, skip=True), _sg(5, "image", 4.0)]
    _md = build_min_durations(_segs)
    assert _md == {"2": 5.0, "10": 7.25} and list(_md) == ["2", "10"], _md
    assert all(isinstance(v, float) for v in _md.values())
    assert build_min_durations([_sg(1, "image", None)]) == {} and build_min_durations([]) == {}
    assert build_min_durations([_sg(1, "video", 3.0, skip=True)]) == {}
    try:
        build_min_durations([_sg(1, "video", None)])
    except ValueError as e:
        assert "Сегмент 1" in str(e) and "min_duration" in str(e), e
    else:
        raise AssertionError("видео без min_duration должно останавливать")
    with tempfile.TemporaryDirectory() as _d:
        _links = os.path.join(_d, "sub", "links.txt")
        os.makedirs(os.path.dirname(_links))
        _pth = write_min_durations(_links, _segs)
        assert _pth == os.path.join(_d, "sub", "min_durations.json"), _pth
        with open(_pth, "r", encoding="utf-8") as fh:
            _raw = fh.read()
        assert json.loads(_raw) == {"2": 5.0, "10": 7.25} and "5.0" in _raw, _raw
        assert os.listdir(os.path.dirname(_pth)) == ["min_durations.json"]  # tmp-файла не осталось
        write_min_durations(_links, [_sg(1, "image", None)])  # нет видео -> пустой объект, файл есть
        with open(_pth, "r", encoding="utf-8") as fh:
            assert json.load(fh) == {}

    # load_requests: min_duration
    _gv = good["1"]
    assert load({"1": _gv})[0].min_duration == 5.0
    assert load({"1": dict(_gv, min_duration=3)})[0].min_duration == 3.0
    assert load({"1": dict(_gv, min_duration=0)})[0].min_duration == 0.0
    # у image и у skip-видео поля может не быть
    _no_md = {k: v for k, v in _gv.items() if k != "min_duration"}
    assert load({"1": dict(_no_md, type="image")})[0].min_duration is None
    assert load({"1": dict(_no_md, skip=True)})[0].min_duration is None
    # видео без min_duration (нет поля / null) - остановка, с перечислением номеров
    for _bad_set in ({"1": _no_md, "7": dict(_no_md, min_duration=None)},):
        try:
            load(_bad_set)
        except ValueError as e:
            assert "min_duration" in str(e) and "1, 7" in str(e), e
        else:
            raise AssertionError("ожидалась ValueError: видео без min_duration")
    for _bad_md in ("5", True, -1, float("nan"), [5]):
        try:
            load({"1": dict(_gv, min_duration=_bad_md)})
        except ValueError as e:
            assert "min_duration" in str(e) and "Сегмент" in str(e), e
        else:
            raise AssertionError("ожидалась ValueError для min_duration=%r" % (_bad_md,))

    # режим 2: пропуск сайтов без видео в белом списке (wikimedia пропущен, остальные нет)
    assert site_skipped_in_video_only("wikimedia") is True
    assert not any(site_skipped_in_video_only(x) for x in ("pexels", "pixabay", "nasa", "loc"))
    assert video_only_skip_message("wikimedia") == "режим 2: сайт wikimedia пропущен, видео не отдаёт"

    # fetch_and_filter: ранний фильтр длительности и пропуск сайта в режиме 2 (без сети)
    import types as _t2

    def _cand(site, cid, dur):
        return Candidate(site=site, cand_id=cid, text="", license_ok=True,
                         preview_url="http://x/p.jpg", page_url="http://x/" + cid, duration=dur)

    _calls = []

    def _mk_fake(site, items):
        async def _fake(ctx_, q_, mt_):
            _calls.append((site, mt_))
            return list(items)
        return _fake

    _sites_backup = dict(SITE_SEARCH_FUNCS)
    SITE_SEARCH_FUNCS["pexels"] = _mk_fake("pexels", [
        _cand("pexels", "short", 3), _cand("pexels", "edge", 9), _cand("pexels", "long", 20),
        _cand("pexels", "nofield", None), _cand("pexels", "zero", 0), _cand("pexels", "junk", "x"),
    ])
    SITE_SEARCH_FUNCS["wikimedia"] = _mk_fake("wikimedia", [_cand("wikimedia", "w1", None)])
    SITE_SEARCH_FUNCS["nasa"] = _mk_fake("nasa", [_cand("nasa", "n1", None)])
    try:
        _vseg = _sg(1, "video", 10.0)
        _ctx = _t2.SimpleNamespace(site_stats={}, media_mode=1)
        _kept = asyncio.run(fetch_and_filter(_ctx, "pexels", _vseg, "q", "medium"))
        assert [c.cand_id for c in _kept] == ["edge", "long", "nofield", "zero", "junk"], _kept
        _st = _ctx.site_stats["pexels"]
        assert _st.duration_rejected_total == 1 and _st.duration_unknown_total == 3, _st
        # у сегмента без min_duration и у фото фильтра нет
        _ctx = _t2.SimpleNamespace(site_stats={}, media_mode=1)
        _k2 = asyncio.run(fetch_and_filter(_ctx, "pexels", _sg(1, "video", None), "q", "medium"))
        assert len(_k2) == 6 and _ctx.site_stats["pexels"].duration_rejected_total == 0
        _k3 = asyncio.run(fetch_and_filter(_ctx, "pexels", _sg(1, "image", 10.0), "q", "medium"))
        assert len(_k3) == 6
        # NASA: длительности в ответе нет, фильтр не применяется
        _k4 = asyncio.run(fetch_and_filter(_ctx, "nasa", _vseg, "q", "medium"))
        assert [c.cand_id for c in _k4] == ["n1"]
        # все отсеяны -> пусто
        SITE_SEARCH_FUNCS["pexels"] = _mk_fake("pexels", [_cand("pexels", "a", 1), _cand("pexels", "b", 2)])
        _ctx = _t2.SimpleNamespace(site_stats={}, media_mode=1)
        assert asyncio.run(fetch_and_filter(_ctx, "pexels", _vseg, "q", "medium")) == []
        # режим 2: wikimedia для видео вообще не вызывается; для фото и в режиме 1 - вызывается
        _calls.clear()
        _ctx = _t2.SimpleNamespace(site_stats={}, media_mode=2)
        assert asyncio.run(fetch_and_filter(_ctx, "wikimedia", _vseg, "q", "medium")) == []
        assert _calls == [] and _ctx.site_stats == {}, (_calls, _ctx.site_stats)
        asyncio.run(fetch_and_filter(_ctx, "wikimedia", _sg(1, "image", None), "q", "medium"))
        assert _calls == [("wikimedia", "image")], _calls
        _ctx = _t2.SimpleNamespace(site_stats={}, media_mode=1)
        asyncio.run(fetch_and_filter(_ctx, "wikimedia", _vseg, "q", "medium"))
        assert _calls[-1] == ("wikimedia", "video"), _calls
        _ctx = _t2.SimpleNamespace(site_stats={}, media_mode=2)
        asyncio.run(fetch_and_filter(_ctx, "nasa", _vseg, "q", "medium"))
        assert _calls[-1] == ("nasa", "video"), _calls
    finally:
        SITE_SEARCH_FUNCS.clear()
        SITE_SEARCH_FUNCS.update(_sites_backup)

    # ---- ЧАСТЬ Б: переключение типа в режиме 1 ----
    assert should_switch_type(1, False) is True
    assert should_switch_type(1, True) is False      # уже принят нормально
    assert should_switch_type(2, False) is False and should_switch_type(3, False) is False
    assert other_media_type("image") == "video" and other_media_type("video") == "image"
    try:
        other_media_type("gif")
    except ValueError as e:
        assert "gif" in str(e), e
    else:
        raise AssertionError("ожидалась ValueError для неизвестного типа")

    def _bs(i, t, md, skip=False, sites=("pexels", "wikimedia")):
        return SegmentSpec(index=i, scene="s", sites=list(sites), query_narrow="a b", query_medium="c d",
                           query_broad=None, type=t, is_entity=False, entity_keywords=[], skip=skip,
                           min_duration=md)
    validate_switch_inputs([_bs(1, "image", 3.0), _bs(2, "video", 5.0), _bs(3, "image", None, skip=True)], 1)
    validate_switch_inputs([_bs(1, "image", None)], 2)   # режимы 2/3 - без проверки
    validate_switch_inputs([_bs(1, "image", None)], 3)
    try:
        validate_switch_inputs([_bs(4, "image", None), _bs(2, "video", 5.0), _bs(9, "image", None)], 1)
    except ValueError as e:
        assert "min_duration" in str(e) and "4, 9" in str(e), e
    else:
        raise AssertionError("ожидалась ValueError: нет min_duration в режиме 1")

    # итоговый тип -> min_durations
    _sw = {1: "video", 2: "image"}
    _fs = final_segments([_bs(1, "image", 4.0), _bs(2, "video", 6.0), _bs(3, "video", 7.0)], _sw)
    assert [x.type for x in _fs] == ["video", "image", "video"]
    assert build_min_durations(_fs) == {"1": 4.0, "3": 7.0}, build_min_durations(_fs)
    _orig = _bs(1, "image", 4.0)
    assert effective_segment(_orig, {}) is _orig and _orig.type == "image"

    # каскад запасного типа: для video wikimedia убирается, для image - нет
    _av = alt_cascade(_bs(1, "video", 4.0))
    assert _av and all("wikimedia" not in v.sites and "pexels" in v.sites for v in _av), _av
    _ai = alt_cascade(_bs(1, "image", 4.0))
    assert _ai and all("wikimedia" in v.sites for v in _ai), _ai
    assert "переключено 3" in summarize_type_switch(3, 2, Counter({"abs": 1, "best_effort": 1}))

    # process_segment_inner с подменой run_variant/finalize_candidate (без сети)
    def _pc(cid, acc, why, sim=0.3):
        return Candidate(site="pexels", cand_id=cid, text="", license_ok=True, preview_url=None,
                         page_url="p" + cid, own_sim=sim, similarity=sim, rank=1, accepted=acc,
                         reject_reason=why, variant="medium", stats_key="pexels", kind="video")

    def _mkctx(mode):
        return Context(
            session=None, pexels_api_key="", pixabay_api_key="", site_semaphores={},
            global_semaphore=asyncio.Semaphore(1), clip_semaphore=asyncio.Semaphore(1),
            used_files_lock=asyncio.Lock(), media_mode=mode,
        )

    _types_seen: list = []

    def _mk_rv(by_type):
        async def _rv(ctx_, seg_, variant_):
            _types_seen.append(seg_.type)
            return list(by_type.get(seg_.type, []))
        return _rv

    async def _fake_fin2(ctx_, cand_):
        return f"https://cdn.example/{cand_.site}/{cand_.cand_id}"

    _old = (_g["run_variant"], _g["finalize_candidate"])
    _g["finalize_candidate"] = _fake_fin2
    try:
        # режим 1: родной image не принят, video принят по abs -> переключение
        _g["run_variant"] = _mk_rv({"image": [_pc("i1", False, "floor", 0.1)], "video": [_pc("v1", True, "abs")]})
        _c = _mkctx(1)
        _r = asyncio.run(process_segment_inner(_c, _bs(5, "image", 4.0, sites=("pexels",))))
        assert _r[0] == "https://cdn.example/pexels/v1", _r
        assert _c.switch_final == {5: "video"} and _c.switch_attempted == {5} and _c.switch_how["abs"] == 1
        # режим 1: родной принят -> другой тип не трогаем
        _types_seen.clear()
        _g["run_variant"] = _mk_rv({"image": [_pc("i1", True, "abs")], "video": [_pc("v1", True, "abs")]})
        _c = _mkctx(1)
        _r = asyncio.run(process_segment_inner(_c, _bs(5, "image", 4.0, sites=("pexels",))))
        assert _r[0].endswith("/i1") and not _c.switch_attempted and set(_types_seen) == {"image"}, _types_seen
        # режимы 2 и 3: переключения нет, best-effort на родном типе
        for _m in (2, 3):
            _types_seen.clear()
            _g["run_variant"] = _mk_rv({"image": [_pc("i1", False, "floor", 0.1)], "video": [_pc("v1", True, "abs")]})
            _c = _mkctx(_m)
            _r = asyncio.run(process_segment_inner(_c, _bs(5, "image", 4.0, sites=("pexels",))))
            assert _r[0].endswith("/i1") and not _c.switch_final and not _c.switch_attempted, (_m, _r)
            assert set(_types_seen) == {"image"}, _types_seen
        # режим 1: никто не принят ни по одному типу -> best-effort родного типа, тип не меняется
        _g["run_variant"] = _mk_rv({"image": [_pc("i1", False, "floor", 0.1)], "video": [_pc("v1", False, "rank_miss", 0.2)]})
        _c = _mkctx(1)
        _r = asyncio.run(process_segment_inner(_c, _bs(5, "image", 4.0, sites=("pexels",))))
        assert _r[0].endswith("/i1") and not _c.switch_final and _c.switch_attempted == {5}, _r
        # режим 1: по родному ничего не оценено, у другого типа только слабые -> best-effort другого типа
        _g["run_variant"] = _mk_rv({"video": [_pc("v1", False, "rank_miss", 0.2)]})
        _c = _mkctx(1)
        _r = asyncio.run(process_segment_inner(_c, _bs(5, "image", 4.0, sites=("pexels",))))
        assert _r[0].endswith("/v1") and _c.switch_final == {5: "video"} and _c.switch_how["best_effort"] == 1, _r
        # режим 1: нигде ничего -> не найдено (missing), переключение не засчитано как найденное
        _g["run_variant"] = _mk_rv({})
        _c = _mkctx(1)
        assert asyncio.run(process_segment_inner(_c, _bs(5, "image", 4.0, sites=("pexels",)))) == (None, None)
        assert _c.switch_attempted == {5} and not _c.switch_final
    finally:
        _g["run_variant"], _g["finalize_candidate"] = _old

    # ---- ключ кандидата (site, kind, cand_id) ----
    def _kc(site, kind, cid):
        return Candidate(site=site, cand_id=cid, text="", license_ok=True, preview_url=None,
                         page_url=None, kind=kind)

    _kp, _kv = _kc("pexels", "photo", "1"), _kc("pexels", "video", "1")
    assert cand_key(_kp) != cand_key(_kv), (cand_key(_kp), cand_key(_kv))
    assert cand_key(_kp) == cand_key(_kc("pexels", "photo", "1")) == ("pexels", "photo", "1")
    assert cand_key(_kc("pixabay", "photo", "1")) != cand_key(_kc("pixabay", "video", "1"))
    assert cand_key(_kp) != cand_key(_kc("pixabay", "photo", "1"))
    assert kind_of_media_type("image") == "photo" and kind_of_media_type("video") == "video"
    for _bad_kind in ("", "image", None):
        try:
            cand_key(_kc("pexels", _bad_kind, "1"))
        except ValueError:
            pass
        else:
            raise AssertionError(f"cand_key принял kind={_bad_kind!r}")
    try:
        kind_of_media_type("gif")
    except ValueError:
        pass
    else:
        raise AssertionError("kind_of_media_type принял gif")
    # used_files: photo id=1 и video id=1 не конфликтуют; повтор того же кандидата - конфликт
    async def _fake_fin3(ctx_, cand_):
        return f"https://cdn.example/{cand_.kind}/{cand_.cand_id}"

    _old_fin3 = _g["finalize_candidate"]
    _g["finalize_candidate"] = _fake_fin3
    try:
        _uctx3 = _types.SimpleNamespace(
            used_files=set(), used_files_lock=asyncio.Lock(), primary_cands={},
            choice_reasons=Counter(), choice_own_sims=[], backup_reasons=Counter(), backup_own_sims=[],
        )
        assert asyncio.run(_claim_first(_uctx3, [_kp]))[0] == "https://cdn.example/photo/1"
        assert asyncio.run(_claim_first(_uctx3, [_kp]))[0] is None
        assert asyncio.run(_claim_first(_uctx3, [_kv]))[0] == "https://cdn.example/video/1"
        assert _uctx3.used_files == {("pexels", "photo", "1"), ("pexels", "video", "1")}
        # _backup_pool: исключение по exclude_key отличает photo от video с тем же id
        async def _fake_faf(ctx_, site_, seg_, query_, name_, stats_key=None):
            return [_kc("pexels", "photo", "7"), _kc("pexels", "video", "7")]

        async def _fake_score(ctx_, top_, **kw):
            return list(top_)

        _old_bp = (_g["fetch_and_filter"], _g["score_candidates"])
        _g["fetch_and_filter"], _g["score_candidates"] = _fake_faf, _fake_score
        try:
            _bp = asyncio.run(_backup_pool(
                _uctx3, "pexels", _types.SimpleNamespace(index=1), _types.SimpleNamespace(query="q", name="n"),
                cand_key(_kc("pexels", "photo", "7")),
            ))
        finally:
            _g["fetch_and_filter"], _g["score_candidates"] = _old_bp
        assert [cand_key(c) for c in _bp] == [("pexels", "video", "7")], _bp
    finally:
        _g["finalize_candidate"] = _old_fin3
    # кэш векторов: один id разных типов не делит вектор
    _cc = EmbeddingCache()

    async def _enc_p():
        return "vec_photo"

    async def _enc_v():
        return "vec_video"

    assert asyncio.run(_cc.get_or_compute(cand_key(_kp), _enc_p)) == "vec_photo"
    assert asyncio.run(_cc.get_or_compute(cand_key(_kv), _enc_v)) == "vec_video"
    assert asyncio.run(_cc.get_or_compute(cand_key(_kp), _enc_v)) == "vec_photo"

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
