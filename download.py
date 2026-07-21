import os
import re
import json
import ssl
import urllib.request
import urllib.error
from urllib.parse import urlparse
import subprocess

OUTPUT_DIR = "downloaded_media"
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

def parse_and_download_links(env_name: str) -> None:
    raw_text = os.environ.get(env_name, "")
    if not raw_text:
        print("Список ссылок пуст. Нечего скачивать.")
        return

    print("Начинаю разбор и скачивание файлов...")
    
    for line in raw_text.strip().split("\n"):
        if not line or ":" not in line:
            continue
        try:
            num_str, url_str = line.split(":", 1)
            number = int(num_str.strip())
            url = url_str.strip()

            pexels_video_id = extract_pexels_video_id(url)
            is_generic_video = url.lower().endswith((".mp4", ".mov", ".avi"))

            if pexels_video_id:
                download_pexels_video(number, pexels_video_id)
            elif is_generic_video:
                download_via_ytdlp(number, url)
            else:
                download_file(number, url)
        except Exception as e:
            print(f"[ОШИБКА] Строка '{line}' -> {e}")

def extract_pexels_video_id(url: str):
    """Достаёт числовой ID видео из ссылки вида pexels.com/video/... /12345/"""
    match = re.search(r"pexels\.com/(?:[a-z-]+/)?video/[^/]*?(\d+)", url.lower())
    if match:
        return match.group(1)
    return None

def download_pexels_video(number: int, video_id: str) -> None:
    """Скачивает видео напрямую через официальный Pexels API — в обход Cloudflare
    и HTML-страницы сайта, которая блокирует раннеры GitHub Actions."""
    if not PEXELS_API_KEY:
        print(f"[ОШИБКА] Видео {number}: нет PEXELS_API_KEY. Добавьте секрет в репозиторий "
              f"(Settings -> Secrets and variables -> Actions) с ключом от https://www.pexels.com/api/")
        return

    api_url = f"https://api.pexels.com/v1/videos/videos/{video_id}"
    request = urllib.request.Request(api_url, headers={"Authorization": PEXELS_API_KEY})

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[ОШИБКА] Pexels API вернул ошибку для видео {number} (id={video_id}): {e.code} {e.reason}")
        return
    except Exception as e:
        print(f"[ОШИБКА] Не удалось обратиться к Pexels API для видео {number}: {e}")
        return

    video_files = data.get("video_files", [])
    if not video_files:
        print(f"[ОШИБКА] У видео {number} (id={video_id}) нет доступных файлов в ответе API")
        return

    # Берём файл с наибольшей шириной (лучшее качество), предпочитая mp4
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
        print(f"[OK] ВИДЕО-ФУТАЖ {number} (Pexels id={video_id}, {best.get('width')}x{best.get('height')}) успешно скачан")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать файл видео {number} по прямой ссылке: {e}")

def download_via_ytdlp(number: int, url: str) -> None:
    """Резервный путь для видео не с Pexels (обычные ссылки на .mp4/.mov и т.п.)."""
    print(f"Запуск yt-dlp с маскировкой под Chrome для видео: {url}")
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
            print(f"[OK] ВИДЕО-ФУТАЖ {number} успешно скачан")
        else:
            print(f"[ОШИБКА] Ошибка запуска yt-dlp для {number}:\n{result.stderr}")
    except Exception as ytdl_err:
        print(f"[ОШИБКА] Не удалось запустить yt-dlp: {ytdl_err}")

def guess_extension(url: str, content_type: str = "") -> str:
    path = urlparse(url).path
    match = re.search(r"\.(jpg|jpeg|png|webp|gif|mp4|mov)$", path, re.IGNORECASE)
    if match:
        return "." + match.group(1).lower()
    if "jpeg" in content_type or "jpg" in content_type: return ".jpg"
    if "png" in content_type: return ".png"
    if "mp4" in content_type: return ".mp4"
    return ".jpg"

def download_file(number: int, url: str) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    request = urllib.request.Request(url, headers=headers)
    context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            content_type = response.headers.get("Content-Type", "")
            ext = guess_extension(url, content_type)
            filename = f"{number}{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(response.read())
            print(f"[OK] КАРТИНКА {number} ({filename}) успешно скачана")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать картинку {number}: {e}")

def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    parse_and_download_links("INPUT_LINKS")

if __name__ == "__main__":
    main()
