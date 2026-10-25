"""Общие константы и правила для config_loader и weights_log.

Вынесено, чтобы убрать циклический импорт: раньше config_loader импортировал speed_threshold_mbps
из weights_log, а weights_log — DEFAULT_MODEL_PROFILE из config_loader.
Модуль не импортирует ничего из пакета Generate.core (только math). Оба старых модуля
реэкспортируют эти имена, поэтому прежние импорты продолжают работать.
"""
import math

# Профиль модели по умолчанию: вход model_profile, запасной стиль и значение по умолчанию в коде.
# Единственное место литерала «ltx25» в коде (config.default.yaml и входы generate.yml задают то же значение файлами).
DEFAULT_MODEL_PROFILE = "ltx25"

# Имя профиля или пресета (имя файла без расширения): латинские буквы, цифры, «_» и «-».
# Один шаблон для config_loader (входы профиля и пресета) и gemini_prompts (файл пресета).
PROFILE_NAME_PATTERN = r"[A-Za-z0-9_\-]+"
# Значение пресета «без пресета»: всегда допустимо, файла не требует.
PRESET_NONE = "none"


def speed_threshold_mbps(min_inet_mbps: float) -> int:
    """Порог правила скорости, Мбит/с: половина min_inet_mbps входа запуска, округлённая вниз до целого.

    Вход — workflow MIN_INET или --min-inet-mbps (cfg.min_inet_mbps), а не значение из конфига.
    Меньше 1 Мбит/с — ошибка входа: проверяется в config_loader (GeminiInputError).
    """
    return math.floor(float(min_inet_mbps) / 2.0)
