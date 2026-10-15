"""Офлайн-тесты постоянного списка плохих машин (SPEC 3, 8.2).

Никакой сети: Vast — заглушка, список — файлы во временном каталоге.
Сообщения на русском (SPEC 0.5); значения секретов сюда не попадают (SPEC 0.1).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import httpx
import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2], _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from Generate.core import bad_hosts as bh
    from Generate.core import orchestrator as orch
    from Generate.core import vast_client as vc
except ImportError:  # pragma: no cover
    from core import bad_hosts as bh  # type: ignore
    from core import orchestrator as orch  # type: ignore
    from core import vast_client as vc  # type: ignore

SEED_ID = 58482
NOW = 1000.0


# ---------------------------------------------------------------------------
# Заглушки
# ---------------------------------------------------------------------------

def make_offer(offer_id: int, machine_id: int, price: float = 0.3):
    return vc.GpuOffer(offer_id=offer_id, gpu_name="RTX 5090", num_gpus=1, vram_gb=32.0,
                       price_per_hr=price, reliability=0.99, inet_down_mbps=3000.0,
                       inet_up_mbps=3000.0, machine_id=machine_id, is_interruptible=True)


class FakeVast:
    """Только поиск и создание карты; запоминает, по каким предложениям была аренда."""

    def __init__(self, offers) -> None:
        self.offers = list(offers)
        self.search_calls = 0
        self.rented_offers: list = []

    def search_offers(self, filters):
        self.search_calls += 1
        return list(self.offers)

    def create_instance(self, offer_id, image, disk_gb, env_vars=None, label=None, onstart=None,
                        bid_price=None):
        self.rented_offers.append(offer_id)
        return 900 + len(self.rented_offers)


def make_orchestrator(tmp_path, vast, excluded=None, **kw):
    """Оркестратор без запуска конвейера: нужен только _rent_one."""
    return orch.Orchestrator(
        [], tmp_path / "missing.txt", None, None, object(), vast, object(),
        tmp_path / "prompts.json", excluded_machines=excluded, work_dir=tmp_path,
        worker_token="test-token", http_client=httpx.Client(), **kw)


# ---------------------------------------------------------------------------
# Чтение файла: пустой список, комментарии, дубли
# ---------------------------------------------------------------------------

def test_missing_file_gives_empty_set(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    assert bh.load_bad_hosts(tmp_path / "нет.txt") == frozenset()
    assert "пуст" in caplog.text


def test_empty_file_gives_empty_set(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    f = tmp_path / "bad_hosts.txt"
    f.write_text("", encoding="utf-8")
    assert bh.load_bad_hosts(f) == frozenset()
    assert "пуст" in caplog.text


def test_comments_blank_lines_inline_comments_and_duplicates(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    f = tmp_path / "bad_hosts.txt"
    f.write_text("# шапка\n\n58482  # cdi\n   \n# только комментарий\n7\n58482\n  12   # а # б\n",
                 encoding="utf-8")
    assert bh.load_bad_hosts(f) == frozenset({58482, 7, 12})
    assert "3 шт." in caplog.text           # дубли сворачиваются: 58482 учтён один раз


def test_utf8_bom_and_crlf_are_accepted(tmp_path):
    f = tmp_path / "bad_hosts.txt"
    f.write_bytes("\ufeff58482\r\n7\r\n".encode("utf-8"))
    assert bh.load_bad_hosts(f) == frozenset({58482, 7})


# ---------------------------------------------------------------------------
# Неверные строки: код выхода 2, номер строки назван, содержимое не выводится
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "-5", "0", "1.5", "58482 58483", "+5"])
def test_invalid_line_stops_with_exit_code_2_and_names_line(tmp_path, bad):
    f = tmp_path / "bad_hosts.txt"
    f.write_text(f"58482\n{bad}\n", encoding="utf-8")
    with pytest.raises(bh.BadHostsError) as info:
        bh.load_bad_hosts(f)
    assert info.value.exit_code == 2
    assert "строка 2" in str(info.value)


def test_all_invalid_lines_are_reported_together(tmp_path):
    f = tmp_path / "bad_hosts.txt"
    f.write_text("x\n1\ny\n", encoding="utf-8")
    with pytest.raises(bh.BadHostsError) as info:
        bh.load_bad_hosts(f)
    assert "строка 1" in str(info.value) and "строка 3" in str(info.value)


def test_invalid_line_content_is_not_echoed(tmp_path):
    """Если в файл по ошибке попадёт что-то похожее на ключ, в ошибку оно не попадает."""
    f = tmp_path / "bad_hosts.txt"
    f.write_text("TESTKEY-12345\n", encoding="utf-8")
    with pytest.raises(bh.BadHostsError) as info:
        bh.load_bad_hosts(f)
    assert "TESTKEY" not in str(info.value)
    assert "строка 1" in str(info.value)


def test_parse_rejects_zero_and_negative_with_line_numbers():
    with pytest.raises(bh.BadHostsError) as info:
        bh.parse_bad_hosts("0\n-1\n")
    assert "строка 1" in str(info.value) and "строка 2" in str(info.value)


# ---------------------------------------------------------------------------
# Путь и содержимое файла-затравки
# ---------------------------------------------------------------------------

def test_default_path_is_next_to_generate_code():
    """Путь строится от кода модуля, не от текущего каталога."""
    assert bh.DEFAULT_BAD_HOSTS_PATH.name == "bad_hosts.txt"
    assert bh.DEFAULT_BAD_HOSTS_PATH.parent == Path(bh.__file__).resolve().parents[1]


def test_seed_file_lists_machine_58482():
    """Затравка из постановки. При изменении списка этот тест нужно обновить."""
    assert SEED_ID in bh.load_bad_hosts()


def test_module_reads_no_environment():
    src = Path(bh.__file__).read_text(encoding="utf-8")
    assert "environ" not in src and "getenv" not in src


# ---------------------------------------------------------------------------
# Оркестратор: предложения с исключённой машиной не арендуются
# ---------------------------------------------------------------------------

def test_offer_of_listed_machine_is_not_rented(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    vast = FakeVast([make_offer(1, SEED_ID)])
    orc = make_orchestrator(tmp_path, vast, excluded={SEED_ID})
    assert orc._rent_one(NOW) == "retry"
    assert vast.rented_offers == []
    assert str(SEED_ID) in caplog.text


def test_offer_with_unlisted_machine_is_still_rented(tmp_path):
    vast = FakeVast([make_offer(1, SEED_ID), make_offer(2, 22)])
    orc = make_orchestrator(tmp_path, vast, excluded={SEED_ID})
    assert orc._rent_one(NOW) == "ok"
    assert vast.rented_offers == [2]


def test_listed_machine_is_dropped_on_every_search_not_only_first(tmp_path):
    vast = FakeVast([make_offer(1, SEED_ID)])
    orc = make_orchestrator(tmp_path, vast, excluded={SEED_ID})
    assert orc._rent_one(NOW) == "retry"
    assert orc._rent_one(NOW + 60) == "retry"
    assert vast.rented_offers == []


def test_drop_is_logged_once_per_run_not_per_offer(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    vast = FakeVast([make_offer(1, SEED_ID), make_offer(2, SEED_ID), make_offer(3, 22)])
    orc = make_orchestrator(tmp_path, vast, excluded={SEED_ID})
    orc._rent_one(NOW)
    orc._rent_one(NOW + 60)
    lines = [r for r in caplog.records if "bad_hosts.txt" in r.getMessage()]
    assert len(lines) == 1                  # лог один на запуск, а не на предложение
    assert vast.rented_offers == [3, 3]     # оба вызова арендуют предложение 3, 58482 не арендован


def test_in_run_bad_machines_still_excluded_with_permanent_list(tmp_path):
    """Существующее правило _bad_machines сохраняется рядом с постоянным списком."""
    vast = FakeVast([make_offer(1, 22), make_offer(2, 33)])
    orc = make_orchestrator(tmp_path, vast, excluded={SEED_ID})
    orc._bad_machines.add(22)
    assert orc._rent_one(NOW) == "ok"
    assert vast.rented_offers == [2]


def test_no_permanent_list_means_unchanged_behaviour(tmp_path):
    vast = FakeVast([make_offer(1, SEED_ID)])
    orc = make_orchestrator(tmp_path, vast)          # excluded_machines не передан
    assert orc._rent_one(NOW) == "ok"
    assert vast.rented_offers == [1]
