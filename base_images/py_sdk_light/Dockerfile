# https://hub.docker.com/r/supervisely/base-py-sdk-light
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
	LANG=C.UTF-8 \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1


# deps of cv2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgeos-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libexiv2-dev \
        libboost-all-dev \
        ffmpeg \
        libmagic-dev \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log


RUN pip install --upgrade pip

RUN pip install \
    xlrd \
    openpyxl \
    "psutil>=5.9.0" \
    cython \
    "Werkzeug<=3.0.6" \
    scikit-video==1.1.11 \
    attrs \
    "requests>=2.27.1"

RUN pip install "setuptools>=70.0,<81"

ARG tag_ref_name

# Install supervisely package with retry logic for PyPI CDN propagation
RUN bash -c 'set -o pipefail && \
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

RUN pip install --upgrade supervisely[apps,aug,agent,versioning]==$tag_ref_name
