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
    psutil>=5.9.0 \
    cython \
    "Werkzeug<=3.0.6" \
    scikit-video==1.1.11 \
    attrs \
    requests>=2.27.1

RUN pip install "setuptools>=70.0,<81"

ARG tag_ref_name

RUN pip install --upgrade supervisely==$tag_ref_name
RUN pip install --upgrade supervisely[apps,aug,agent]==$tag_ref_name
