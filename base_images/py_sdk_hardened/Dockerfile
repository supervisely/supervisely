# ============================================================================
# STAGE 1: Builder Stage
# ============================================================================
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
RUN pip install setuptools==80.10.1 wheel==0.46.2 pillow==12.1.1
RUN pip install pip-audit==2.7.3

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

# todo: remove; tests only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

# strip installer tooling from the venv
RUN rm -rf \
    /opt/venv/lib/python3.10/site-packages/pip* \
    /opt/venv/lib/python3.10/site-packages/wheel* \
    /opt/venv/bin/pip \
    /opt/venv/bin/pip3 \
    /opt/venv/bin/pip3.* \
    /opt/venv/bin/easy_install \
    /opt/venv/bin/easy_install-* \
    /opt/venv/bin/wheel

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM python:3.10-slim-bookworm as runtime

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# python=python3 symlink
RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/local/bin/python3 /usr/bin/python

# install deps of cv2 and other runtime deps
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

# strip runtime tools
RUN rm -rf /var/lib/dpkg/status \
    /var/lib/dpkg/status.d \
    /var/lib/dpkg/info \
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
    /usr/bin/wget

# remove packaging tooling
RUN rm -rf \
    /usr/local/lib/python3.10/site-packages/pip* \
    /usr/local/lib/python3.10/site-packages/setuptools* \
    /usr/local/lib/python3.10/site-packages/wheel* \
    /usr/local/lib/python3.10/site-packages/pkg_resources* \
    /usr/local/lib/python3.10/site-packages/_distutils* \
    /usr/local/lib/python3.10/site-packages/distutils-precedence.pth