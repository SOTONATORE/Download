# Сборка из корня репозитория: docker build -f Generate/worker/Dockerfile -t videogen-worker .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    COMFY_OUTPUT_DIR=/workspace/ComfyUI/output
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI \
    && pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt \
    && mkdir -p /workspace/ComfyUI/output

RUN pip install --no-cache-dir httpx

RUN mkdir -p /workspace/Generate/worker \
    && touch /workspace/Generate/__init__.py /workspace/Generate/worker/__init__.py
COPY Generate/worker/server.py Generate/worker/comfy_client.py Generate/worker/setup.sh /workspace/Generate/worker/
RUN sed -i 's/\r$//' /workspace/Generate/worker/setup.sh \
    && chmod +x /workspace/Generate/worker/setup.sh

# 8000 — HTTP API воркера (для раннера), 8188 — ComfyUI (только внутри контейнера).
EXPOSE 8000 8188

ENTRYPOINT ["/workspace/Generate/worker/setup.sh"]
