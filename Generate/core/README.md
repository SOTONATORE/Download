# core/ — ядро генерации

Логика второй части без привязки к GitHub Actions: разбор `final.srt` и `missing.txt`, расчёт длины клипа (`timing.py`), промпты Gemini (`gemini_prompts.py`, `gemini_orchestrator.py`), клиент Vast.ai (`vast_client.py`), учёт бюджета (`budget.py`), оркестратор запуска (`orchestrator.py`) и выходной адаптер GitHub Release (`release_adapter.py`).

Точка входа — `cli.py` в родительской папке, а не модули этой папки. Ключи API читаются только из окружения. Подробности и правила — в `Generate/SPEC.md` (разделы 3, 6–9).

Здесь нет кода воркера на карте (он в `../worker/`), тестов (в `../tests/`) и диагностических скриптов (в `../tools/`).
