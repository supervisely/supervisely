# ============================================================================
# STAGE 1: Builder Stage
# ============================================================================
FROM python:3.14-slim-bookworm AS builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG tag_ref_name

# Install supervisely package with retry logic for PyPI CDN propagation & cleanup
RUN set -eu; \
    python -m venv /opt/venv; \
    /opt/venv/bin/pip install --upgrade pip==26.0.1; \
    /opt/venv/bin/pip install setuptools==80.10.1 wheel==0.46.2 pillow==12.1.1; \
    # MAX_ATTEMPTS=8; \
    # TIMEOUT=15; \
    # ATTEMPT=1; \
    # while [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; do \
    # echo "Attempt ${ATTEMPT}/${MAX_ATTEMPTS}: Installing supervisely==$tag_ref_name..."; \
    # if /opt/venv/bin/pip install --no-cache-dir supervisely==$tag_ref_name; then \
    # echo "Supervisely installed successfully"; \
    # break; \
    # fi; \
    # if [ "$ATTEMPT" -eq "$MAX_ATTEMPTS" ]; then \
    # echo "ERROR: Failed to install supervisely==$tag_ref_name after $MAX_ATTEMPTS attempts" >&2; \
    # exit 1; \
    # fi; \
    # echo "Installation failed. Waiting ${TIMEOUT}s..."; \
    # sleep "$TIMEOUT"; \
    # ATTEMPT=$((ATTEMPT + 1)); \
    # done; \
    # todo: remove; tests only ------------------------------------- \
    apt-get update; \
    apt-get install -y --no-install-recommends git; \
    rm -rf /var/lib/apt/lists/*; \
    /opt/venv/bin/pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images; \
    # --------------------------------------------------------------- \
    rm -rf \
    /opt/venv/lib/python3.14/site-packages/wheel* \
    /opt/venv/lib/python3.14/site-packages/setuptools/_vendor/wheel* \
    /opt/venv/bin/pip \
    /opt/venv/bin/pip3 \
    /opt/venv/bin/pip3.* \
    /opt/venv/bin/easy_install \
    /opt/venv/bin/easy_install-* \
    /opt/venv/bin/wheel

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM python:3.14-slim-bookworm AS runtime

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# python=python3 symlink
RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/local/bin/python3 /usr/bin/python

# install deps of cv2 and other runtime deps and strip package managers and build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libmagic1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb1 \
    libx11-6 \
    libgeos-c1v5 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    /usr/bin/apt \
    /usr/bin/apt-get \
    /usr/bin/apt-cache \
    /usr/bin/apt-mark \
    /usr/bin/apt-key \
    /usr/bin/dpkg \
    /usr/bin/dpkg-deb \
    /usr/bin/dpkg-query \
    /usr/bin/dpkg-divert \
    /usr/bin/git \
    /usr/bin/curl \
    /usr/bin/wget \
    /usr/local/lib/python3.14/site-packages/pip* \
    /usr/local/lib/python3.14/site-packages/setuptools* \
    /usr/local/lib/python3.14/site-packages/wheel* \
    /usr/local/lib/python3.14/site-packages/pkg_resources* \
    /usr/local/lib/python3.14/site-packages/_distutils* \
    /usr/local/lib/python3.14/site-packages/distutils-precedence.pth