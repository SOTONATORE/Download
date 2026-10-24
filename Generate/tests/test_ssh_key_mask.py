"""F4: маска многострочного VAST_SSH_KEY (core/log_setup.py, secret_mask_values; set_known_secrets).

Правило: маскируется всё значение и каждая его непустая строка длиной не меньше 20 символов; короче — не маскируется;
пустое значение ничего не добавляет. Строки ключа в тестах фальшивые (сгенерированы для проверки маски).
"""
try:
    from Generate.core import log_setup as ls
except ImportError:  # pragma: no cover - плоский импорт
    from core import log_setup as ls  # type: ignore

LONG_LINE = "QfT7kZ2pLw9XcM4nRb8dYh3sGj6aVe1u0Pq5Wi"         # 38 символов
MEDIUM_LINE = "Hk4Rt9Vx2Nz7Lb3Qc8Ym"                          # 21 символ
SHORT_LINE = "ab12Cd34"                                       # 8 символов (меньше 20)
KEY_BEGIN = "-----BEGIN FAKE TEST KEY-----"                   # 28 символов: >= 20, маскируется как строка
FAKE_KEY = "\n".join([KEY_BEGIN, LONG_LINE, SHORT_LINE, MEDIUM_LINE, "-----END FAKE TEST KEY-----"])


def _reset():
    ls.set_known_secrets([])


def test_multiline_key_masks_whole_value_and_long_lines_only():
    try:
        ls.set_known_secrets(ls.secret_mask_values(FAKE_KEY))
        # все длинные строки ключа маскируются
        assert LONG_LINE not in ls.mask_text(f"строка ключа: {LONG_LINE}")
        assert MEDIUM_LINE not in ls.mask_text(f"строка ключа: {MEDIUM_LINE}")
        # всё значение целиком (многострочная форма, как попадает в текст целиком) маскируется
        whole = ls.mask_text(f"ключ:\n{FAKE_KEY}\nконец")
        assert LONG_LINE not in whole and MEDIUM_LINE not in whole and KEY_BEGIN not in whole
        # короткая строка ключа в обычной строке журнала не меняется
        assert ls.mask_text(f"проверка {SHORT_LINE} готова") == f"проверка {SHORT_LINE} готова"
    finally:
        _reset()


def test_empty_key_adds_nothing():
    assert ls.secret_mask_values("") == []
    assert ls.secret_mask_values(None) == []
    assert ls.secret_mask_values("   \n  ") == []
    try:
        ls.set_known_secrets(ls.secret_mask_values(""))         # пустое значение не передаётся маске
        line = "обычная строка журнала без секретов, 12345"
        assert ls.mask_text(line) == line
        assert "" not in ls._KNOWN_SECRETS                      # пустая строка не попала в список замен
    finally:
        _reset()


def test_line_threshold_is_exactly_twenty_characters():
    twenty = "A" * 20
    nineteen = "B" * 19
    values = ls.secret_mask_values("\n".join([twenty, nineteen]))
    assert twenty in values and nineteen not in values
    assert ls.secret_mask_values("x" * 5) == ["x" * 5]          # однострочное короткое значение: только целиком


def test_short_key_line_in_normal_log_line_is_not_changed():
    try:
        ls.set_known_secrets(ls.secret_mask_values(FAKE_KEY))
        assert ls.mask_text(f"Веса: этап: {SHORT_LINE}, 5 Мбит/с") == f"Веса: этап: {SHORT_LINE}, 5 Мбит/с"
    finally:
        _reset()
