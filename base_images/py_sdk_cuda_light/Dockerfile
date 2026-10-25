# https://hub.docker.com/r/supervisely/base-py-sdk-cuda-light
ARG tag_ref_name

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS cuda

FROM python:3.12-slim-bookworm

ARG tag_ref_name

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    LD_LIBRARY_PATH="/usr/lib:/usr/local/lib" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufft.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusolver.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusparse.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvJitLink.so* /usr/lib/

# Keep only the runtime libraries required by the SDK's default dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgeos-c1v5 \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        libsm6 \
        libx11-6 \
        libxcb1 \
        libxext6 \
        libxrender1 \
        libssl3 \
        openssl \
    && rm -rf /var/lib/apt/lists/*

# Pip is intentionally retained for apps that install framework-specific CUDA wheels.
RUN set -eu; \
    python -m pip install --no-cache-dir --upgrade pip==26.1.2; \
    max_attempts=8; \
    timeout=15; \
    attempt=1; \
    while [ "$attempt" -le "$max_attempts" ]; do \
        echo "Attempt ${attempt}/${max_attempts}: Installing supervisely==${tag_ref_name}..."; \
        if python -m pip install --no-cache-dir "supervisely==${tag_ref_name}"; then \
            exit 0; \
        fi; \
        if [ "$attempt" -eq "$max_attempts" ]; then \
            echo "ERROR: Failed to install supervisely==${tag_ref_name} after ${max_attempts} attempts" >&2; \
            exit 1; \
        fi; \
        echo "Installation failed. Waiting ${timeout}s..."; \
        sleep "$timeout"; \
        attempt=$((attempt + 1)); \
    done
