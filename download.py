import os
import re
import ssl
import urllib.request
from urllib.parse import urlparse

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
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    context = ssl._create_unverified_context()

    # === УМНЫЙ ПАРСИНГ ПРЯМОЙ ССЫЛКИ PEXELS ===
    if is_video and "pexels.com" in url:
        try:
            print(f"Парсинг страницы Pexels для поиска видео-файла: {url}")
            request_page = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request_page, timeout=15, context=context) as p_response:
                html = p_response.read().decode('utf-8', errors='ignore')
                # Ищем CDN-ссылки на mp4 файлы прямо в коде страницы
                video_urls = re.findall(r'(https://video-files\.pexels\.com/[^\s"\'<>\\\]+\.mp4)', html)
                if video_urls:
                    url = video_urls[0]
                    print(f"Найдена прямая ссылка на видео: {url}")
                else:
                    # Запасной вариант
                    video_id = url.rstrip("/").split("/")[-1].split("-")[-1]
                    url = f"https://www.pexels.com/video/{video_id}/download"
        except Exception as scrap_err:
            print(f"Ошибка при парсинге страницы Pexels: {scrap_err}")
            video_id = url.rstrip("/").split("/")[-1].split("-")[-1]
            url = f"https://www.pexels.com/video/{video_id}/download"

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            content_type = response.headers.get("Content-Type", "")
            ext = guess_extension(url, content_type)
            if is_video and ext == ".jpg": ext = ".mp4"

            filename = f"{number}{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(response.read())
            print(f"[OK] Скачан {'ВИДЕО-ФУТАЖ' if is_video else 'КАРТИНКА'} {number} ({filename})")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать {number}: {e}")

def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    parse_and_download_links("INPUT_LINKS")

if __name__ == "__main__":
    main()
