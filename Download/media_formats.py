"""Единый белый список форматов медиа: ЛЮБОЙ новый формат добавляется ТОЛЬКО здесь."""

from __future__ import annotations

import threading
from urllib.parse import urlsplit

PHOTO_EXTENSIONS = frozenset({"jpg", "jpeg", "png"})
VIDEO_EXTENSIONS = frozenset({"mp4", "mov", "avi"})
ALLOWED_EXTENSIONS = PHOTO_EXTENSIONS | VIDEO_EXTENSIONS

# Фото-контейнеры ISO BMFF (HEIC/AVIF): тоже имеют "ftyp", но это не видео.
_IMAGE_FTYP_BRANDS = frozenset({b"heic", b"heix", b"heim", b"heis", b"hevm",
                                b"hevs", b"mif1", b"msf1", b"avif", b"avis"})

__all__ = [
    "PHOTO_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "ALLOWED_EXTENSIONS",
    "ext_from_name",
    "is_allowed_ext",
    "sniff_format",
    "describe_rejected",
    "kind_of_ext",
    "FormatRejectStats",
]


def ext_from_name(url_or_title: str) -> str:
    """Расширение в нижнем регистре без точки; для URL учитывается только path."""
    s = (url_or_title or "").strip()
    if "://" in s or s.startswith("//"):
        s = urlsplit(s).path
    else:
        # Название файла ("File:Foo.JPG"): на всякий случай отрезаем query/fragment
        s = s.split("#", 1)[0].split("?", 1)[0]
    name = s.replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    ext = name.rsplit(".", 1)[-1].strip().lower()
    return ext if ext.isalnum() else ""


def is_allowed_ext(ext: str, kind: str | None = None) -> bool:
    """Проверка расширения (без точки, регистр не важен) по белому списку."""
    e = (ext or "").lower().lstrip(".")
    if kind == "photo":
        return e in PHOTO_EXTENSIONS
    if kind == "video":
        return e in VIDEO_EXTENSIONS
    return e in ALLOWED_EXTENSIONS


def sniff_format(data: bytes) -> str | None:
    """Формат по содержимому: jpg/png/mp4/mov/avi или None."""
    if not data or len(data) < 12:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[4:8] == b"ftyp":
        if data[8:12] in _IMAGE_FTYP_BRANDS:
            return None
        return "mov" if data[8:12] == b"qt  " else "mp4"
    if data[0:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "avi"
    return None


def describe_rejected(data: bytes) -> str:
    """Название неподдерживаемого формата для лога."""
    d = data or b""
    if d[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if d[:4] == b"RIFF" and d[8:12] == b"WEBP":
        return "webp"
    if d[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if d[:4] == b"%PDF":
        return "pdf"
    if d[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if d[:4] == b"OggS":
        return "ogg"
    if d[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"):
        return "mpeg"
    head = d[:64].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        return "html"
    if d[4:8] == b"ftyp":
        return "avif" if d[8:12] in (b"avif", b"avis") else "heic"
    return "unknown"


def kind_of_ext(ext: str) -> str | None:
    """'photo' / 'video' / None по расширению."""
    e = (ext or "").lower().lstrip(".")
    if e in PHOTO_EXTENSIONS:
        return "photo"
    if e in VIDEO_EXTENSIONS:
        return "video"
    return None


class FormatRejectStats:
    """Потокобезопасный счётчик отсева по формату: сайт -> расширение -> число."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, int]] = {}

    def add(self, site: str, ext: str) -> None:
        e = (ext or "").lower().lstrip(".") or "unknown"
        with self._lock:
            per_site = self._data.setdefault(site, {})
            per_site[e] = per_site.get(e, 0) + 1

    def total(self) -> int:
        with self._lock:
            return sum(sum(v.values()) for v in self._data.values())

    def summary_line(self) -> str:
        with self._lock:
            if not self._data:
                return ""
            parts = []
            for site in sorted(self._data):
                exts = self._data[site]
                inner = ", ".join(f"{e}: {exts[e]}" for e in sorted(exts))
                parts.append(f"{site} {{{inner}}}")
            return "Отсеяно по формату: " + "; ".join(parts)


if __name__ == "__main__":
    pad = b"\x00" * 16

    # sniff_format: проходят
    assert sniff_format(b"\xff\xd8\xff\xe0" + pad) == "jpg"
    assert sniff_format(b"\x89PNG\r\n\x1a\n" + pad) == "png"
    assert sniff_format(b"\x00\x00\x00\x18ftypmp42" + pad) == "mp4"
    assert sniff_format(b"\x00\x00\x00\x14ftypqt  " + pad) == "mov"
    assert sniff_format(b"RIFF\x00\x00\x00\x00AVI LIST" + pad) == "avi"

    # sniff_format + describe_rejected: отклоняются
    rejected = {
        "gif": b"GIF89a" + pad,
        "webp": b"RIFF\x00\x00\x00\x00WEBPVP8 " + pad,
        "tiff": b"II*\x00" + pad,
        "pdf": b"%PDF-1.7" + pad,
        "webm": b"\x1a\x45\xdf\xa3" + pad,
        "ogg": b"OggS" + pad,
        "mpeg": b"\x00\x00\x01\xba" + pad,
        "html": b"  \n<!DOCTYPE html>" + pad,
        "unknown": b"garbage-bytes-here!",
    }
    for name, blob in rejected.items():
        assert sniff_format(blob) is None, name
        assert describe_rejected(blob) == name, name
    assert describe_rejected(b"MM\x00*" + pad) == "tiff"
    assert describe_rejected(b"GIF87a" + pad) == "gif"
    assert describe_rejected(b"\x00\x00\x01\xb3" + pad) == "mpeg"
    assert describe_rejected(b"<HTML>" + pad) == "html"
    assert sniff_format(b"\x00\x00\x00\x18ftypheic" + pad) is None
    assert describe_rejected(b"\x00\x00\x00\x18ftypheic" + pad) == "heic"
    assert describe_rejected(b"\x00\x00\x00\x1cftypavif" + pad) == "avif"
    assert sniff_format(b"\xff\xd8\xff") is None  # < 12 байт
    assert sniff_format(b"") is None

    # ext_from_name
    assert ext_from_name("https://x.org/a/b/photo.JPG?w=100&f=png#frag") == "jpg"
    assert ext_from_name("File:A.JPG") == "jpg"
    assert ext_from_name("https://x.org/a/b/noext") == ""
    assert ext_from_name("https://x.org/dir.v1/clip") == ""
    assert ext_from_name("") == ""

    # is_allowed_ext / kind_of_ext
    assert is_allowed_ext("JPG") and is_allowed_ext("mp4")
    assert is_allowed_ext("png", "photo") and not is_allowed_ext("png", "video")
    assert is_allowed_ext("mov", "video") and not is_allowed_ext("mov", "photo")
    assert not is_allowed_ext("gif") and not is_allowed_ext("")
    assert kind_of_ext("JPEG") == "photo"
    assert kind_of_ext("avi") == "video"
    assert kind_of_ext("gif") is None

    # FormatRejectStats
    st = FormatRejectStats()
    assert st.total() == 0 and st.summary_line() == ""
    for _ in range(2):
        st.add("wikimedia", "pdf")
    st.add("wikimedia", "svg")
    st.add("nasa", "tif")
    st.add("nasa", "")
    assert st.total() == 5
    assert st.summary_line() == (
        "Отсеяно по формату: nasa {tif: 1, unknown: 1}; wikimedia {pdf: 2, svg: 1}"
    )

    print("ok")
