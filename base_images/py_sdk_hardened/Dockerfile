FROM python:3.10-slim-bookworm AS builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1 wheel==0.46.2

ARG tag_ref_name

# Install supervisely package with retry logic for PyPI CDN propagation
RUN sh -c 'set -eu && \
    MAX_ATTEMPTS=8 && \
    TIMEOUT=15 && \
    ATTEMPT=0 && \
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do \
    echo "Attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS: Installing supervisely==$tag_ref_name..." && \
    if pip install --no-cache-dir --upgrade supervisely==$tag_ref_name; then \
    echo "Supervisely installed successfully" && \
    exit 0; \
    fi && \
    ATTEMPT=$((ATTEMPT + 1)) && \
    if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then \
    echo "Installation failed. Waiting ${TIMEOUT}s..." && \
    sleep $TIMEOUT; \
    fi; \
    done && \
    echo "ERROR: Failed to install supervisely==$tag_ref_name after $MAX_ATTEMPTS attempts" >&2 && \
    exit 1'

# RUN pip install --upgrade supervisely[apps,aug,agent,versioning]==$tag_ref_name
# todo: remove; tests only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

RUN pip install --upgrade \
    "urllib3>=2.6.3" \
    "python-multipart>=0.0.22" \
    "protobuf>=6.33.5" \
    "starlette>=0.49.1" \
    "pillow>=12.1.1"

FROM python:3.10-slim-bookworm

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /var/lib/dpkg/status \
    /var/lib/dpkg/status.d \
    /var/lib/dpkg/info

RUN /opt/venv/bin/python -m pip install --no-cache-dir --upgrade \
    setuptools==78.1.1 \
    wheel==0.46.2 \
    "jaraco.context>=6.1.0" \
    && rm -rf \
    /opt/venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.context* \
    /opt/venv/lib/python3.10/site-packages/setuptools/_vendor/wheel* \
    /usr/local/lib/python3.10/site-packages/pip* \
    /usr/local/lib/python3.10/site-packages/setuptools* \
    /usr/local/lib/python3.10/site-packages/wheel* \
    /usr/local/lib/python3.10/site-packages/pkg_resources* \
    /usr/local/lib/python3.10/site-packages/_distutils* \
    /usr/local/lib/python3.10/site-packages/distutils-precedence.pth
