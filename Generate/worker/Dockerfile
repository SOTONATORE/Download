# Generate/worker/Dockerfile

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl aria2 ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ComfyUI и его зависимости
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI \
    && pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt

RUN pip install --no-cache-dir httpx

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    COMFY_OUTPUT_DIR=/workspace/ComfyUI/output \
    COMFY_URL=http://127.0.0.1:8188 \
    WORKER_HTTP_PORT=8000 \
    WORKER_WORKFLOW_PATH=/workspace/Generate/model_profiles/workflows/ltx25.json \
    WORKER_BINDINGS_PATH=/workspace/Generate/model_profiles/workflows/ltx25_bindings.json

# Файлы воркера и профили моделей
COPY Generate/worker/server.py Generate/worker/comfy_client.py Generate/worker/setup.sh /workspace/Generate/worker/
COPY Generate/model_profiles/ /workspace/Generate/model_profiles/

RUN sed -i 's/\r$//' /workspace/Generate/worker/setup.sh \
    && chmod +x /workspace/Generate/worker/setup.sh

EXPOSE 8000

ENTRYPOINT ["/workspace/Generate/worker/setup.sh"]
