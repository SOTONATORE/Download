import os
import re
import json
import urllib.parse
from urllib.parse import urlparse
import subprocess
from curl_cffi import requests as cffi_requests

OUTPUT_DIR = "downloaded_media"
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
COVERR_API_KEY = os.environ.get("COVERR_API_KEY", "")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
}


def parse_and_download_links(env_name: str) -> None:
    raw_text = os.environ.get(env_name, "")
    if not raw_text:
        print("Список ссылок пуст. Нечего скачивать.")
        return

    print("Начинаю разбор и скачивание файлов...")
    matches = re.findall(r'(\d+)\s*:\s*(https?://[^\s]+)', raw_text)

    if not matches:
        print("[ОШИБКА] Не удалось распознать ссылки формата 'номер: ссылка'")
        return

    print(f"Найдено ссылок для скачивания: {len(matches)}")

    for num_str, url in matches:
        number = int(num_str)
        download_media_item(number, url)


def extract_id(url: str) -> str:
    match = re.search(r"(\d+)/?$", url.strip().rstrip("/"))
    return match.group(1) if match else ""


def extract_og_video_url(html: str) -> str:
    """Достаём прямую ссылку на видео из og:video / og:video:secure_url meta-тега страницы."""
    match = re.search(
        r'<meta[^>]+property=["\']og:video(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if match:
        return match.group(1)
    # На случай, если порядок атрибутов в теге обратный (content раньше property)
    match = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:video(?::secure_url)?["\']',
        html, re.IGNORECASE
    )
    return match.group(1) if match else ""


def looks_like_html(content: bytes) -> bool:
    """Проверяем, не скачали ли мы случайно HTML-страницу вместо медиафайла."""
    snippet = content[:512].lstrip().lower()
    return snippet.startswith(b"<!doctype") or snippet.startswith(b"<html") or b"<head>" in snippet[:1024]


def download_media_item(number: int, url: str) -> None:
    url = url.strip()
    url_lower = url.lower()

    # 1. PEXELS ВИДЕО
    if "pexels.com" in url_lower and "/video/" in url_lower:
        video_id = extract_id(url)
        if video_id:
            download_pexels_video(number, video_id)
        return

    # 2. PEXELS ФОТО
    if "pexels.com" in url_lower and "/photo/" in url_lower:
        photo_id = extract_id(url)
        if photo_id and PEXELS_API_KEY:
            download_pexels_photo(number, photo_id)
        return

    # 3. PIXABAY (ФОТО И ВИДЕО ЧЕРЕЗ ОФИЦИАЛЬНЫЙ API)
    if "pixabay.com" in url_lower:
        item_id = extract_id(url)
        if item_id and PIXABAY_API_KEY:
            if "/videos/" in url_lower:
                download_pixabay_video(number, item_id)
            else:
                download_pixabay_photo(number, item_id)
            return

    # 4. COVERR ВИДЕО (ТОЧНАЯ ССЫЛКА ИЗ og:video НА САМОЙ СТРАНИЦЕ, БЕЗ API/ПОИСКА)
    if "coverr.co" in url_lower:
        download_coverr_video(number, url)
        return

    # 5. LIBRARY OF CONGRESS (loc.gov) - ЧЕРЕЗ ОФИЦИАЛЬНЫЙ JSON API
    if "loc.gov" in url_lower:
        download_loc_gov(number, url)
        return

    # 6. ПРЯМЫЕ ССЫЛКИ НА ВИДЕОФАЙЛЫ + СТРАНИЦЫ MIXKIT -> ЧЕРЕЗ yt-dlp
    if url_lower.endswith((".mp4", ".mov", ".avi")) or "mixkit.co" in url_lower:
        download_via_ytdlp(number, url)
        return

    # 7. ВСЁ ОСТАЛЬНОЕ - СКАЧИВАНИЕ ЧЕРЕЗ curl_cffi С ИМИТАЦИЕЙ БРАУЗЕРА
    download_direct_via_cffi(number, url)


def download_direct_via_cffi(number: int, url: str) -> None:
    try:
        resp = cffi_requests.get(url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        content = resp.content

        if looks_like_html(content):
            print(f"[ОШИБКА] {number}: похоже, скачалась HTML-страница, а не медиафайл ({url})")
            return

        content_type = resp.headers.get("Content-Type", "")
        ext = guess_extension(url, content_type)
        filename = f"{number}{ext}"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(content)
        print(f"[OK] ФАЙЛ {number} ({filename}) успешно скачан")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать {number}: {e}")


def download_pixabay_photo(number: int, photo_id: str) -> None:
    api_url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&id={photo_id}"
    try:
        resp = cffi_requests.get(api_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            direct_url = hits[0].get("largeImageURL") or hits[0].get("imageURL")
            if direct_url:
                img_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
                img_resp.raise_for_status()
                filepath = os.path.join(OUTPUT_DIR, f"{number}.jpg")
                with open(filepath, "wb") as f:
                    f.write(img_resp.content)
                print(f"[OK] ФОТО PIXABAY {number} (id={photo_id}) успешно скачано")
                return
        print(f"[ОШИБКА] Не удалось получить фото Pixabay {number}")
    except Exception as e:
        print(f"[ОШИБКА] Pixabay API ошибка {number}: {e}")


def download_pixabay_video(number: int, video_id: str) -> None:
    api_url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&id={video_id}"
    try:
        resp = cffi_requests.get(api_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            videos = hits[0].get("videos", {})
            best_video = videos.get("large") or videos.get("medium") or videos.get("small")
            if best_video and "url" in best_video:
                direct_url = best_video["url"]
                vid_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
                vid_resp.raise_for_status()
                filepath = os.path.join(OUTPUT_DIR, f"{number}.mp4")
                with open(filepath, "wb") as f:
                    f.write(vid_resp.content)
                print(f"[OK] ВИДЕО PIXABAY {number} (id={video_id}) успешно скачано")
                return
        print(f"[ОШИБКА] Не удалось получить видео Pixabay {number}")
    except Exception as e:
        print(f"[ОШИБКА] Pixabay API ошибка {number}: {e}")


def download_pexels_photo(number: int, photo_id: str) -> None:
    api_url = f"https://api.pexels.com/v1/photos/{photo_id}"
    try:
        resp = cffi_requests.get(api_url, headers={**BROWSER_HEADERS, "Authorization": PEXELS_API_KEY},
                                  impersonate="chrome", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        direct_url = data.get("src", {}).get("original") or data.get("src", {}).get("large")
        if direct_url:
            img_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
            img_resp.raise_for_status()
            filepath = os.path.join(OUTPUT_DIR, f"{number}.jpg")
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            print(f"[OK] ФОТО PEXELS {number} (id={photo_id}) успешно скачано")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка скачивания фото Pexels {number}: {e}")


def download_pexels_video(number: int, video_id: str) -> None:
    if not PEXELS_API_KEY:
        print(f"[ОШИБКА] Видео {number}: нет PEXELS_API_KEY.")
        return

    api_url = f"https://api.pexels.com/v1/videos/videos/{video_id}"
    try:
        resp = cffi_requests.get(api_url, headers={**BROWSER_HEADERS, "Authorization": PEXELS_API_KEY},
                                  impersonate="chrome", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ОШИБКА] Pexels API ошибка {number}: {e}")
        return

    video_files = data.get("video_files", [])
    if not video_files:
        print(f"[ОШИБКА] Pexels {number}: нет video_files в ответе API")
        return

    mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4"]
    candidates = mp4_files if mp4_files else video_files
    best = max(candidates, key=lambda f: f.get("width") or 0)
    direct_url = best["link"]

    try:
        resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
        resp.raise_for_status()
        filepath = os.path.join(OUTPUT_DIR, f"{number}.mp4")
        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"[OK] ВИДЕО PEXELS {number} (id={video_id}) успешно скачано")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать файл видео {number}: {e}")


def download_coverr_video(number: int, url: str) -> None:
    try:
        page_resp = cffi_requests.get(url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
        page_resp.raise_for_status()
        html = page_resp.text

        direct_url = extract_og_video_url(html)
        if not direct_url:
            print(f"[ОШИБКА] Coverr {number}: не нашли og:video на странице {url}")
            return

        vid_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
        vid_resp.raise_for_status()
        content = vid_resp.content

        if looks_like_html(content):
            print(f"[ОШИБКА] Coverr {number}: по ссылке из og:video пришла HTML-страница, а не видео")
            return

        filepath = os.path.join(OUTPUT_DIR, f"{number}.mp4")
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"[OK] ВИДЕО COVERR {number} успешно скачано ({direct_url})")
    except Exception as e:
        print(f"[ОШИБКА] Coverr ошибка {number}: {e}")


def download_loc_gov(number: int, url: str) -> None:
    # Официальный JSON API Библиотеки Конгресса не требует ключа - достаточно добавить fo=json
    parsed = urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query["fo"] = ["json"]
    json_url = parsed._replace(query=urllib.parse.urlencode(query, doseq=True)).geturl()

    try:
        resp = cffi_requests.get(json_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        data = resp.json()

        direct_url = None
        resource = data.get("resource", {}) or {}
        # Пытаемся достать самое качественное изображение из разных возможных мест ответа
        if isinstance(resource.get("files"), list):
            for file_group in resource["files"]:
                if isinstance(file_group, list) and file_group:
                    candidate = max(file_group, key=lambda f: f.get("height", 0) if isinstance(f, dict) else 0)
                    if isinstance(candidate, dict) and candidate.get("url"):
                        direct_url = candidate["url"]
                        break
        if not direct_url:
            item = data.get("item", {}) or {}
            image_url = item.get("image_url")
            if isinstance(image_url, list) and image_url:
                direct_url = image_url[-1]
            elif isinstance(image_url, str):
                direct_url = image_url

        if not direct_url:
            print(f"[ОШИБКА] loc.gov {number}: не удалось найти прямую ссылку на файл в JSON-ответе")
            return

        file_resp = cffi_requests.get(direct_url, headers=BROWSER_HEADERS, impersonate="chrome", timeout=60)
        file_resp.raise_for_status()
        content = file_resp.content

        if looks_like_html(content):
            print(f"[ОШИБКА] loc.gov {number}: похоже, скачалась HTML-страница, а не файл")
            return

        ext = guess_extension(direct_url, file_resp.headers.get("Content-Type", ""))
        filepath = os.path.join(OUTPUT_DIR, f"{number}{ext}")
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"[OK] LOC.GOV {number} успешно скачан")
    except Exception as e:
        print(f"[ОШИБКА] loc.gov ошибка {number}: {e}")


def download_via_ytdlp(number: int, url: str) -> None:
    try:
        cmd = [
            "yt-dlp",
            "--extractor-args", "generic:impersonate=chrome",
            "-o", f"{OUTPUT_DIR}/{number}.%(ext)s",
            "--format", "mp4/best",
            "--no-check-certificate",
            "--retries", "3",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] ВИДЕО {number} успешно скачано через yt-dlp")
        else:
            print(f"[ОШИБКА] Ошибка yt-dlp {number}: {result.stderr}")
    except Exception as ytdl_err:
        print(f"[ОШИБКА] Не удалось запустить yt-dlp: {ytdl_err}")


def guess_extension(url: str, content_type: str = "") -> str:
    path = urlparse(url).path
    match = re.search(r"\.(jpg|jpeg|png|webp|gif|mp4|mov)$", path, re.IGNORECASE)
    if match:
        return "." + match.group(1).lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "mp4" in content_type:
        return ".mp4"
    return ".jpg"


def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parse_and_download_links("INPUT_LINKS")


if __name__ == "__main__":
    main()
