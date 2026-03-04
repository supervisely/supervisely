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
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip==26.0.1
RUN pip install setuptools==80.10.1 wheel==0.46.2 pillow==12.1.1
RUN pip install pip-audit==2.7.3

ARG tag_ref_name

# Install supervisely package with retry logic for PyPI CDN propagation
# RUN sh -c 'set -eu && \
#     MAX_ATTEMPTS=8 && \
#     TIMEOUT=15 && \
#     ATTEMPT=0 && \
#     while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do \
#     echo "Attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS: Installing supervisely==$tag_ref_name..." && \
#     if pip install --no-cache-dir --upgrade supervisely==$tag_ref_name; then \
#     echo "Supervisely installed successfully" && \
#     exit 0; \
#     fi && \
#     ATTEMPT=$((ATTEMPT + 1)) && \
#     if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then \
#     echo "Installation failed. Waiting ${TIMEOUT}s..." && \
#     sleep $TIMEOUT; \
#     fi; \
#     done && \
#     echo "ERROR: Failed to install supervisely==$tag_ref_name after $MAX_ATTEMPTS attempts" >&2 && \
#     exit 1'

# todo: remove; tests only -------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images
RUN pip install pillow==12.1.1 wheel==0.46.2
# ---------------------------------------------------------------

# strip installer tooling from the venv
RUN rm -rf \
    /opt/venv/lib/python3.14/site-packages/wheel* \
    /opt/venv/lib/python3.14/site-packages/setuptools/_vendor/wheel* \
    /opt/venv/bin/pip \
    /opt/venv/bin/pip3 \
    /opt/venv/bin/pip3.* \
    /opt/venv/bin/easy_install \
    /opt/venv/bin/easy_install-* \
    /opt/venv/bin/wheel

# ============================================================================
# STAGE 2: System Deps Stage
# ============================================================================
FROM python:3.14-slim-bookworm AS system_deps

# Runtime deps required by CV2/shapely/ffmpeg stack
ARG SYS_PACKAGES="libgeos-dev libglib2.0-0 libxext6 libsm6 libxrender1 libgl1 libmagic1 libxcb1 libx11-6 ffmpeg"

# Install system dependencies and copy to separate stage for later use
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    mkdir -p /opt/system-rootfs /tmp/debs; \
    apt-get update; \
    apt-get install -y --no-install-recommends --download-only $SYS_PACKAGES; \
    cp /var/cache/apt/archives/*.deb /tmp/debs/; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*; \
    for deb in /tmp/debs/*.deb; do \
    pkg="$(dpkg-deb -f "$deb" Package)"; \
    case "$pkg" in libc6|libc-bin|libc6-dev) continue ;; esac; \
    dpkg-deb -x "$deb" /opt/system-rootfs; \
    done; \
    mkdir -p /opt/system-rootfs/usr /opt/system-rootfs/etc /opt/system-rootfs/lib /opt/system-rootfs/lib64; \
    rm -rf /tmp/debs

RUN mkdir -p /opt/system-rootfs/usr/local/bin /opt/system-rootfs/usr/bin \
    && ln -sf /opt/python/bin/python3 /opt/system-rootfs/usr/local/bin/python3 \
    && ln -sf /opt/python/bin/python3 /opt/system-rootfs/usr/local/bin/python3.14 \
    && ln -sf /opt/python/bin/python3 /opt/system-rootfs/usr/local/bin/python \
    && ln -sf /opt/python/bin/python3 /opt/system-rootfs/usr/bin/python3 \
    && ln -sf /opt/python/bin/python3 /opt/system-rootfs/usr/bin/python

# ============================================================================
# STAGE 3: Runtime Stage (Hardened)
# ============================================================================
FROM dhi.io/python:3.14 AS runtime

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=system_deps /opt/system-rootfs/usr/ /usr/
COPY --from=system_deps /opt/system-rootfs/etc/ /etc/
COPY --from=system_deps /opt/system-rootfs/lib/ /usr/lib/
COPY --from=system_deps /opt/system-rootfs/lib64/ /usr/lib64/
ENV PATH="/opt/venv/bin:$PATH"