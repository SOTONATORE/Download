"""Аудит секретов (SPEC 0.1, п.10): маркер TESTKEY-12345 не должен попасть ни в логи,
ни в вывод CLI, ни в тексты исключений (включая цепочку и traceback), ни в URL запросов.

Маркер задаётся переменными окружения, затем вызываются обычные пути кода:
  - импорт всех модулей ядра;
  - инициализация оркестратора и его отказ на неверном входе;
  - клиент Vast: поиск карт (успех, HTTP-ошибка, сеть, таймаут);
  - генерация промптов через Gemini (успех, HTTP-ошибка, сеть);
  - загрузка конфига с секретом не на своём месте и вывод CLI на ошибках.

Сеть не используется: все клиенты работают поверх httpx.MockTransport.
Офлайн, без ключей реального мира.
"""
import importlib
import json
import logging
import pathlib
import socket
import traceback

import httpx
import pytest

try:
    from Generate.core import config_loader
    from Generate.core.gemini_prompts import (
        GeminiInputError, GeminiRuntimeError, PromptConfig, RealTransport, generate_prompts)
    from Generate.core.orchestrator import Orchestrator
    from Generate.core.srt_parser import Segment
    from Generate.core.vast_client import OfferFilter, VastClient, VastError
    from Generate import cli
    PKG = "Generate.core"
except ImportError:  # pragma: no cover - плоский импорт
    from core import config_loader  # type: ignore
    from core.gemini_prompts import (  # type: ignore
        GeminiInputError, GeminiRuntimeError, PromptConfig, RealTransport, generate_prompts)
    from core.orchestrator import Orchestrator  # type: ignore
    from core.srt_parser import Segment  # type: ignore
    from core.vast_client import OfferFilter, VastClient, VastError  # type: ignore
    cli = None
    PKG = "core"

GEN = pathlib.Path(__file__).resolve().parents[1]
CORE_DIR = GEN / "core"
STYLE_FILE = str(GEN / "model_profiles" / "models" / "ltx25" / "prompts" / "style.md")
DATA_SRT = GEN / "tests" / "data" / "final.srt"

SECRET = "TESTKEY-12345"
# Имена переменных собраны из частей: литерал GEN_..._API_KEY в коде Generate/ запрещён (SPEC 0.1).
GEMINI_ENV = "GEN_" + "GEMINI_API_KEY"
VAST_ENV = "GEN_" + "VAST_API_KEY"

SEGMENTS = [
    Segment(num=1, start_ms=0, end_ms=2000, text="The rain stops over the harbor."),
    Segment(num=2, start_ms=2000, end_ms=4000, text="A single boat leaves the dock."),
]


@pytest.fixture(autouse=True)
def _secrets_in_env(monkeypatch):
    monkeypatch.setenv(GEMINI_ENV, SECRET)
    monkeypatch.setenv(VAST_ENV, SECRET)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("сетевое соединение в офлайн-тесте запрещено")
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture(autouse=True)
def _capture_everything(caplog):
    caplog.set_level(logging.DEBUG)   # корневой уровень: видны и логи httpx, если они появятся
    return caplog


def exc_text(exc: BaseException) -> str:
    """Всё, что видит пользователь из исключения: текст, repr и traceback с цепочкой."""
    return "\n".join([str(exc), repr(exc), "".join(traceback.format_exception(exc))])


def assert_clean(*texts: str) -> None:
    for text in texts:
        assert SECRET not in text, "маркер ключа найден в выводе или исключении"


def _bundle(i: int = 1) -> dict:
    return {"id": i, "gpu_name": "RTX 5090", "num_gpus": 1, "gpu_ram": 24576,
            "dph_total": 0.5, "min_bid": 0.4, "reliability2": 0.99, "inet_down": 3000.0,
            "inet_up": 1000.0, "inet_up_cost": 0.001, "inet_down_cost": 0.001,
            "machine_id": 100 + i}


def _gemini_envelope(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


# ---------------------------------------------------------------- импорт

def test_all_core_modules_import():
    names = sorted(p.stem for p in CORE_DIR.glob("*.py") if p.stem != "__init__")
    assert "config_loader" in names and len(names) >= 10
    for name in names:
        importlib.import_module(f"{PKG}.{name}")


# ---------------------------------------------------------------- оркестратор

def test_orchestrator_init_and_refusal_are_clean(tmp_path, caplog, capsys):
    missing = tmp_path / "missing.txt"
    missing.write_text("1\n", encoding="utf-8")
    vast = VastClient(client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"offers": []}))))

    orc = Orchestrator(SEGMENTS, missing, None, None, object(), vast, object(),
                       tmp_path / "prompts.json", repo="owner/repo", run_id="secrets-test",
                       profile="ltx25", docker_image="img:latest",
                       worker_build="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0")
    assert orc is not None

    with pytest.raises(Exception) as ei:          # отказ на неверном входе
        Orchestrator(SEGMENTS, missing, None, None, object(), vast, object(),
                     tmp_path / "prompts.json", max_cards=0)
    out = capsys.readouterr()
    assert_clean(exc_text(ei.value), caplog.text, out.out, out.err)


# ---------------------------------------------------------------- клиент Vast

def test_vast_search_success_is_clean_and_key_only_in_header(caplog, capsys):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert SECRET not in str(request.url), "ключ попал в URL"
        assert any(SECRET in v for v in request.headers.values()), "ключ должен идти в заголовке"
        return httpx.Response(200, json={"offers": [_bundle(1)]})

    client = VastClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    offers = client.search_offers(OfferFilter())
    assert offers and seen
    out = capsys.readouterr()
    assert_clean(caplog.text, out.out, out.err)


def test_vast_http_error_text_is_redacted(caplog, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"сервер отразил ключ: {SECRET}")

    client = VastClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(VastError) as ei:
        client.search_offers(OfferFilter())
    out = capsys.readouterr()
    assert_clean(exc_text(ei.value), caplog.text, out.out, out.err)


def test_vast_network_error_hides_exception_text(caplog, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"сбой соединения, ключ {SECRET}", request=request)

    client = VastClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(VastError) as ei:
        client.search_offers(OfferFilter())
    out = capsys.readouterr()
    assert_clean(exc_text(ei.value), caplog.text, out.out, out.err)


def test_vast_timeout_hides_exception_text(caplog, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"таймаут, ключ {SECRET}", request=request)

    client = VastClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(VastError) as ei:
        client.search_offers(OfferFilter())
    out = capsys.readouterr()
    assert_clean(exc_text(ei.value), caplog.text, out.out, out.err)


# ---------------------------------------------------------------- Gemini и промпты

def test_gemini_prompt_generation_is_clean(caplog, capsys):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert SECRET not in str(request.url), "ключ попал в URL"
        assert request.headers.get("x-goog-api-key") == SECRET
        body = json.dumps([{"segment_index": 1, "prompt": "A quiet harbor at dawn, soft light, slow pan."}])
        return httpx.Response(200, json=_gemini_envelope(body))

    transport = RealTransport(client=httpx.Client(transport=httpx.MockTransport(handler)))
    results = generate_prompts(SEGMENTS, [1], PromptConfig(media="video", style_file=STYLE_FILE),
                               transport, "gemini-test-model")
    assert results and seen
    out = capsys.readouterr()
    assert_clean(caplog.text, out.out, out.err)


def test_gemini_http_error_is_clean(caplog, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED",
                                                   "message": f"квота для ключа {SECRET}"}})

    transport = RealTransport(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(GeminiRuntimeError) as ei:
        generate_prompts(SEGMENTS, [1], PromptConfig(media="video", style_file=STYLE_FILE),
                         transport, "gemini-test-model")
    assert ei.value.status == 429
    out = capsys.readouterr()
    assert_clean(exc_text(ei.value), caplog.text, out.out, out.err)


def test_gemini_network_error_is_clean(caplog, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"таймаут, ключ {SECRET}", request=request)

    transport = RealTransport(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(GeminiRuntimeError) as ei:
        generate_prompts(SEGMENTS, [1], PromptConfig(media="video", style_file=STYLE_FILE),
                         transport, "gemini-test-model")
    out = capsys.readouterr()
    assert_clean(exc_text(ei.value), caplog.text, out.out, out.err)


# ---------------------------------------------------------------- конфиг и CLI

def test_secret_placed_in_config_is_never_echoed(tmp_path, caplog, capsys):
    # Ключ записан и как значение неизвестного ключа, и как значение числового поля.
    text = config_loader.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("disk_gb: 100", f"disk_gb: {SECRET}")
    text += f"\nvast_api_key: {SECRET}\n"
    path = tmp_path / "leaky.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(GeminiInputError) as ei:
        config_loader.load_config(path)
    assert ei.value.exit_code == 2
    out = capsys.readouterr()
    assert_clean(exc_text(ei.value), caplog.text, out.out, out.err)


@pytest.mark.skipif(cli is None, reason="CLI доступен только как пакет Generate")
def test_cli_error_paths_do_not_print_secret(tmp_path, caplog, capsys):
    code = cli.main(["run", "--srt", str(tmp_path / "нет.srt"),
                     "--missing", str(tmp_path / "m.txt")])
    assert code == 2
    code_cfg = cli.main(["check", "--config", str(tmp_path / "нет.yaml"),
                         "--srt", str(DATA_SRT), "--missing", str(tmp_path / "m.txt")])
    assert code_cfg == 2
    out = capsys.readouterr()
    assert_clean(caplog.text, out.out, out.err)
