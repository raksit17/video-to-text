# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib

WORKDIR /app

# CTranslate2 needs the OpenMP runtime. PyAV wheels decode media without an FFmpeg executable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# Opt-in GPU image: Linux CUDA 12 cuBLAS/cuDNN 9 wheels, not Windows GPU packages.
ARG INSTALL_GPU=false
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        python -m pip install \
            'nvidia-cublas-cu12>=12,<13' \
            'nvidia-cudnn-cu12>=9,<10' \
            'nvidia-cuda-runtime-cu12>=12,<13'; \
    fi

COPY video_transcriber.py web_app.py ./
COPY web/ ./web/

RUN useradd --system --uid 10001 --create-home --home-dir /home/app app \
    && mkdir -p /app/.work "$HF_HOME" \
    && chown -R app:app /app/.work /home/app/.cache

USER app
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=4).close()" || exit 1

# The app defaults to loopback on bare metal; inside Docker it must bind to all interfaces.
CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "8765"]
