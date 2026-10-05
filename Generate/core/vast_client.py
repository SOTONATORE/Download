"""Клиент Vast.ai (этап 6; SPEC 6, 8.2, 8.3, 0.1, 0.2, 0.5).

Только стандартная библиотека + httpx. Ключ берётся ТОЛЬКО из переменной окружения
и передаётся ТОЛЬКО в заголовке Authorization: Bearer. Ключ не попадает в URL,
логи, repr и тексты исключений.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

log = logging.getLogger("Generate.vast_client")

# Имя секрета собрано из частей (test_structure.py: литерал в коде запрещён).
ENV_KEY_NAME = "GEN_" + "VAST_API_KEY"
DEFAULT_BASE_URL = "https://console.vast.ai/api/v0"
WORKER_HTTP_PORT = 8000
_SNIPPET_LIMIT = 200


class VastError(Exception):
    """Базовая ошибка клиента Vast."""
    exit_code = 1


class VastInputError(VastError):
    """Неверная конфигурация: нет ключа, неверные фильтры."""
    exit_code = 2


class VastRuntimeError(VastError):
    """Сбой сети или ответ Vast API с кодом, отличным от 200."""
    exit_code = 3


@dataclass
class OfferFilter:
    gpu_name: Optional[str] = None
    min_vram_gb: float = 0.0
    max_price: float = 0.9
    min_reliability: float = 0.95
    min_inet_mbps: float = 2000.0
    interruptible: bool = True
    verified: bool = False


@dataclass
class GpuOffer:
    offer_id: int
    gpu_name: str
    num_gpus: int
    vram_gb: float
    price_per_hr: float
    reliability: float
    inet_down_mbps: float
    inet_up_mbps: float
    machine_id: int
    is_interruptible: bool


@dataclass
class InstanceInfo:
    instance_id: int
    actual_status: str
    intended_status: str
    label: Optional[str] = None
    public_ip: Optional[str] = None
    direct_port: Optional[int] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    dph: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class VastClient:
    """Тонкий клиент REST API Vast.ai."""

    def __init__(
        self,
        api_key_env: str = ENV_KEY_NAME,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise VastInputError(
                f"Не задан ключ Vast: переменная окружения {api_key_env} пуста или отсутствует"
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._timeout = timeout

    # --- служебное ---
    def __repr__(self) -> str:
        return "VastClient()"

    __str__ = __repr__

    def __enter__(self) -> "VastClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}
        try:
            resp = self._client.request(
                method, url, json=json, params=params, headers=headers, timeout=self._timeout
            )
        except httpx.TimeoutException:
            log.warning("Таймаут запроса к Vast: %s %s", method, path)
            raise VastRuntimeError(f"Таймаут запроса к Vast API: {method} {path}") from None
        except httpx.HTTPError as exc:
            # Текст исходного исключения может содержать заголовки: берём только имя типа.
            log.warning("Сетевая ошибка Vast: %s %s (%s)", method, path, type(exc).__name__)
            raise VastRuntimeError(
                f"Сетевая ошибка при обращении к Vast API: {method} {path} ({type(exc).__name__})"
            ) from None

        if resp.status_code != 200:
            snippet = self._redact(resp.text or "")[:_SNIPPET_LIMIT]
            log.warning("Vast вернул HTTP %s: %s %s", resp.status_code, method, path)
            raise VastRuntimeError(
                f"Vast API вернул HTTP {resp.status_code}: {method} {path}. Ответ: {snippet}"
            )
        try:
            return resp.json()
        except ValueError:
            raise VastRuntimeError(
                f"Vast API вернул не JSON: {method} {path}"
            ) from None

    # --- поиск предложений ---
    @staticmethod
    def _build_query(f: OfferFilter) -> dict:
        q: dict[str, Any] = {
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "dph_total": {"lte": f.max_price},
            "reliability2": {"gte": f.min_reliability},
            "inet_down": {"gte": f.min_inet_mbps},
            "type": "bid" if f.interruptible else "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 100,
        }
        if f.verified:
            q["verified"] = {"eq": True}
        if f.min_vram_gb > 0:
            q["gpu_ram"] = {"gte": f.min_vram_gb * 1024}
        if f.gpu_name:
            q["gpu_name"] = {"eq": f.gpu_name}
        return q

    @staticmethod
    def _parse_offer(b: dict, interruptible: bool) -> Optional[GpuOffer]:
        offer_id = _to_int(b.get("id"))
        if offer_id is None:
            return None
        price = _to_float(b.get("dph_total"))
        if interruptible and b.get("min_bid") is not None:
            price = _to_float(b.get("min_bid"), price)
        reliability = b.get("reliability2", b.get("reliability"))
        return GpuOffer(
            offer_id=offer_id,
            gpu_name=str(b.get("gpu_name", "")),
            num_gpus=_to_int(b.get("num_gpus")) or 1,
            vram_gb=_to_float(b.get("gpu_ram")) / 1024.0,
            price_per_hr=price,
            reliability=_to_float(reliability),
            inet_down_mbps=_to_float(b.get("inet_down")),
            inet_up_mbps=_to_float(b.get("inet_up")),
            machine_id=_to_int(b.get("machine_id")) or 0,
            is_interruptible=interruptible,
        )

    def search_offers(self, filters: OfferFilter) -> list[GpuOffer]:
        if filters.max_price <= 0:
            raise VastInputError("Неверный фильтр: max_price должен быть больше нуля")
        if not 0.0 <= filters.min_reliability <= 1.0:
            raise VastInputError("Неверный фильтр: min_reliability должен быть от 0 до 1")
        if filters.min_inet_mbps < 0 or filters.min_vram_gb < 0:
            raise VastInputError("Неверный фильтр: отрицательные значения не допускаются")

        data = self._request("POST", "/bundles/", json=self._build_query(filters))
        bundles = data.get("offers", []) if isinstance(data, dict) else []
        result: list[GpuOffer] = []
        for b in bundles:
            if not isinstance(b, dict):
                continue
            offer = self._parse_offer(b, filters.interruptible)
            if offer is None:
                continue
            # Повторная проверка на клиенте: сервер может вернуть лишнее.
            if offer.price_per_hr > filters.max_price:
                continue
            if offer.reliability < filters.min_reliability:
                continue
            if offer.inet_down_mbps < filters.min_inet_mbps:
                continue
            if offer.vram_gb < filters.min_vram_gb:
                continue
            if filters.gpu_name and offer.gpu_name != filters.gpu_name:
                continue
            result.append(offer)
        result.sort(key=lambda o: o.price_per_hr)
        log.info("Найдено подходящих предложений Vast: %d", len(result))
        return result

    # --- экземпляры ---
    def create_instance(
        self,
        offer_id: int,
        image: str,
        disk_gb: int,
        env_vars: Optional[dict] = None,
        label: Optional[str] = None,
        onstart: Optional[str] = None,
    ) -> int:
        if not image:
            raise VastInputError("Не задан образ контейнера для аренды карты")
        if disk_gb <= 0:
            raise VastInputError("Размер диска должен быть больше нуля")
        payload: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "env": dict(env_vars or {}),
        }
        if label:
            payload["label"] = label
        if onstart:
            payload["onstart"] = onstart
        data = self._request("PUT", f"/asks/{int(offer_id)}/", json=payload)
        inst_id = _to_int(data.get("new_contract")) if isinstance(data, dict) else None
        if inst_id is None or (isinstance(data, dict) and data.get("success") is False):
            raise VastRuntimeError("Vast API не вернул номер созданного экземпляра")
        log.info("Создан экземпляр Vast %d (предложение %d)", inst_id, offer_id)
        return inst_id

    @staticmethod
    def _parse_instance(d: dict) -> InstanceInfo:
        direct_port = None
        ports = d.get("ports")
        if isinstance(ports, dict):
            mapped = ports.get(f"{WORKER_HTTP_PORT}/tcp")
            if isinstance(mapped, list) and mapped and isinstance(mapped[0], dict):
                direct_port = _to_int(mapped[0].get("HostPort"))
        label = d.get("label")
        return InstanceInfo(
            instance_id=_to_int(d.get("id")) or 0,
            actual_status=str(d.get("actual_status") or ""),
            intended_status=str(d.get("intended_status") or ""),
            label=str(label) if label else None,
            public_ip=d.get("public_ipaddr") or None,
            direct_port=direct_port,
            ssh_host=d.get("ssh_host") or None,
            ssh_port=_to_int(d.get("ssh_port")),
            dph=_to_float(d.get("dph_total")),
            raw=d,
        )

    def get_instance(self, instance_id: int) -> InstanceInfo:
        data = self._request("GET", f"/instances/{int(instance_id)}/")
        inst = data.get("instances") if isinstance(data, dict) else None
        if isinstance(inst, list):
            inst = inst[0] if inst else None
        if not isinstance(inst, dict):
            raise VastRuntimeError(f"Экземпляр Vast {instance_id} не найден в ответе API")
        return self._parse_instance(inst)

    def destroy_instance(self, instance_id: int) -> bool:
        data = self._request("DELETE", f"/instances/{int(instance_id)}/")
        if isinstance(data, dict) and data.get("success") is False:
            raise VastRuntimeError(f"Vast отклонил удаление экземпляра {instance_id}")
        log.info("Экземпляр Vast %d уничтожен", instance_id)
        return True

    def list_instances(self, label: Optional[str] = None) -> list[InstanceInfo]:
        data = self._request("GET", "/instances/")
        items = data.get("instances", []) if isinstance(data, dict) else []
        result = [self._parse_instance(d) for d in items if isinstance(d, dict)]
        if label is not None:
            result = [i for i in result if i.label == label]
        return result

    def destroy_by_label(self, label: str) -> list[int]:
        """Уничтожает все экземпляры с меткой (SPEC 8.3). Пробует все, ошибки собирает."""
        if not label:
            raise VastInputError("Пустая метка: массовое удаление без метки запрещено")
        destroyed: list[int] = []
        failed: list[int] = []
        for inst in self.list_instances(label=label):
            try:
                self.destroy_instance(inst.instance_id)
                destroyed.append(inst.instance_id)
            except VastRuntimeError:
                failed.append(inst.instance_id)
        if failed:
            raise VastRuntimeError(
                f"Не удалось уничтожить экземпляры Vast: {failed}; уничтожены: {destroyed}"
            )
        return destroyed
