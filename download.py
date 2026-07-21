import os
import re
import ssl
import urllib.request
from urllib.parse import urlparse

OUTPUT_DIR = "downloaded_media"

# 1. Сюда вставляй ссылки на КАРТИНКИ от Клода
# (Формат -> номер: "ссылка",)
IMAGES = {
    35: "https://example.com/photo_35.jpg",
    36: "https://example.com/photo_36.jpg",
}

# 2. Сюда вставляй ссылки на ВИДЕО с Pexels от Клода
# (Формат -> номер: "ссылка",)
BROLL_VIDEOS = {
    30: "https://www.pexels.com/video/7947467",
    31: "https://www.pexels.com/video/6120119",
}

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
    if is_video and "pexels.com" in url:
        # Надежно извлекаем ID видео из ссылки любого формата
        clean_url = url.rstrip("/")
        video_id = clean_url.split("/")[-1].split("-")[-1]
        url = f"https://www.pexels.com/video/{video_id}/download"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    request = urllib.request.Request(url, headers=headers)
    
    # Отключаем строгую проверку SSL во избежание ошибок на сервере
    context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            content_type = response.headers.get("Content-Type", "")
            ext = guess_extension(url, content_type)
            if is_video and ext == ".jpg": ext = ".mp4"

            filename = f"{number}{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(response.read())
            print(f"[OK] Скачан файл {number} ({filename})")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось скачать {number}: {e}")

def main():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Начинаю скачивание файлов...")
    for num, url in IMAGES.items():
        if "example.com" not in url: download_file(num, url, is_video=False)
    for num, url in BROLL_VIDEOS.items():
        download_file(num, url, is_video=True)

if __name__ == "__main__":
    main()
