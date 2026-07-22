import os
import re
import json
import ssl
import urllib.request
import urllib.error
from urllib.parse import urlparse
import subprocess
import cloudscraper

OUTPUT_DIR = "downloaded_media"
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

def parse_and_download_links(env_name: str) -> None:
    raw_text = os.environ.get(env_name, "")
    if not raw_text:
        print("Список ссылок пуст. Нечего скачивать.")
        return

    print("Начинаю разбор и скачивание файлов...")
    
    # Регулярное выражение находит ВСЕ пары "номер: ссылка", даже если они вставлены в одну строку через пробел
    matches = re.findall(r'(\d+)\s*:\s*(https?://[^\s]+)', raw_text)
    
    if not matches:
        print("[ОШИБКА] Не удалось распознать ссылки формата 'номер: ссылка'")
        return

    print(f"Найдено ссылок для скачивания: {len(matches)}")

    for num_str, url in matches:
        number = int(num_str)
        download_media_item(number, url)

def download_media_item(number: int, url: str) -> None:
    url = url.strip()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    context = ssl._create_unverified_context()

    # 1. PEXELS ВИДЕО
    pexels_vid = re.search(r"pexels\.com/(?:[a-z-]+/)?video/[^/]*?(\d+)", url.lower())
    if pexels_vid:
        download_pexels_video(number, pexels_vid.group(1))
        return

    # 2. PEXELS ФОТО
    pexels_img = re.search(r"pexels\.com/(?:[a-z-]+/)?photo/[^/]*?(\d+)", url.lower())
    if pexels_img and PEXELS_API_KEY:
        download_pexels_photo(number, pexels_img.group(1))
        return

    # 3. PIXABAY ИЛИ СТРАНИЦЫ ФОТОСТОКОВ
    if "pixabay.com" in url.lower() or not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov")):
        try:
            scraper = cloudscraper.create_scraper()
            html = scraper.get(url, timeout=15).text
            img_urls = re.findall(r'(https://cdn\.pixabay\.com/[^"\']+\.(?:jpg|png|webp))', html)
            if img_urls:
                url = img_urls[0].replace(r"\/", "/").replace("\\/", "/")
        except Exception:
            pass

    # 4. ОБЫЧНЫЕ ВИДЕО (.mp4)
    if url.lower().endswith((".mp4", ".mov", ".avi")):
        download_via_ytdlp(number, url)
        return

    # 5. СТАНДАРТНОЕ СКАЧИВАНИЕ ФОТО
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=context) as response:
            content_type = response.headers.get("Content-Type", "")
            ext = guess_extension(url, content_type)
            filename = f"{number}{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(response.read())
            print(f"[OK] КАРТИНКА {number} ({filename}) успешно скачана")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать {number}: {e}")

def download_pexels_photo(number: int, photo_id: str) -> None:
    api_url = f"https://api.pexels.com/v1/photos/{photo_id}"
    req = urllib.request.Request(api_url, headers={"Authorization": PEXELS_API_KEY, "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            direct_url = data.get("src", {}).get("original") or data.get("src", {}).get("large")
            if direct_url:
                img_req = urllib.request.Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(img_req, timeout=30) as img_resp:
                    filepath = os.path.join(OUTPUT_DIR, f"{number}.jpg")
                    with open(filepath, "wb") as f:
                        f.write(img_resp.read())
                print(f"[OK] ФОТО PEXELS {number} (id={photo_id}) успешно скачано")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать фото Pexels {number}: {e}")

def download_pexels_video(number: int, video_id: str) -> None:
    if not PEXELS_API_KEY:
        print(f"[ОШИБКА] Видео {number}: нет PEXELS_API_KEY.")
        return

    api_url = f"https://api.pexels.com/v1/videos/videos/{video_id}"
    request = urllib.request.Request(
        api_url,
        headers={
            "Authorization": PEXELS_API_KEY,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[ОШИБКА] Pexels API ошибка {number}: {e}")
        return

    video_files = data.get("video_files", [])
    if not video_files:
        print(f"[ОШИБКА] Нет доступных файлов для видео {number}")
        return

    mp4_files = [f for f in video_files if f.get("file_type") == "video/mp4"]
    candidates = mp4_files if mp4_files else video_files
    best = max(candidates, key=lambda f: f.get("width") or 0)
    direct_url = best["link"]

    try:
        req = urllib.request.Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            filepath = os.path.join(OUTPUT_DIR, f"{number}.mp4")
            with open(filepath, "wb") as f:
                f.write(response.read())
        print(f"[OK] ВИДЕО PEXELS {number} (id={video_id}) успешно скачано")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать файл видео {number}: {e}")

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
    if match: return "." + match.group(1).lower()
    if "jpeg" in content_type or "jpg" in content_type: return ".jpg"
    if "png" in content_type: return ".png"
    if "mp4" in content_type: return ".mp4"
    return ".jpg"

def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    parse_and_download_links("INPUT_LINKS")

if __name__ == "__main__":
    main()
