import os
import re
import ssl
import urllib.request
from urllib.parse import urlparse
import subprocess

OUTPUT_DIR = "downloaded_media"

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
            
            is_video = "pexels.com/video" in url.lower() or url.lower().endswith((".mp4", ".mov", ".avi"))
            
            download_file(number, url, is_video=is_video)
        except Exception as e:
            print(f"[ОШИБКА] Строка '{line}' -> {e}")

def guess_extension(url: str, content_type: str = "") -> str:
    path = urlparse(url).path
    match = re.search(r"\.(jpg|jpeg|png|webp|gif|mp4|mov)$", path, re.IGNORECASE)
    if match:
        return "." + match.group(1).lower()
    if "jpeg" in content_type or "jpg" in content_type: return ".jpg"
    if "png" in content_type: return ".png"
    if "mp4" in content_type: return ".mp4"
    return ".jpg"

def download_file(number: int, url: str, is_video: bool = False) -> None:
    # === ЕСЛИ ЭТО ВИДЕО — КАЧАЕМ ЧЕРЕЗ YT-DLP С МАСКИРОВКОЙ ПОД CHROME ===
    if is_video:
        print(f"Запуск yt-dlp с маскировкой под Chrome для видео: {url}")
        try:
            # Запускаем консольную команду yt-dlp с обходом Cloudflare
            cmd = [
                "yt-dlp",
                "--extractor-args", "generic:impersonate",
                "-o", f"{OUTPUT_DIR}/{number}.%(ext)s",
                "--format", "mp4/best",
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[OK] ВИДЕО-ФУТАЖ {number} успешно скачан")
            else:
                print(f"[ОШИБКА] Ошибка запуска yt-dlp: {result.stderr}")
        except Exception as ytdl_err:
            print(f"[ОШИБКА] Не удалось запустить yt-dlp: {ytdl_err}")
        return

    # === ЕСЛИ ЭТО КАРТИНКА — КАЧАЕМ СТАНДАРТНО ===
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
