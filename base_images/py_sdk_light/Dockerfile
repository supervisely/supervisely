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
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log


RUN pip install --upgrade pip

RUN apt-get update && \
    apt-get -y install \
        libexiv2-dev \
        libboost-all-dev

# to support pandas.read_excel
RUN pip install xlrd
RUN pip install psutil>=5.9.0
RUN pip install cython

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install Werkzeug==1.0.1
RUN pip install scikit-video==1.1.11


RUN pip install attrs
RUN pip install requests>=2.27.1
RUN apt-get install -y libmagic-dev

RUN pip install --upgrade setuptools==70.3.0

ARG tag_ref_name

RUN pip install --upgrade supervisely==$tag_ref_name
RUN pip install --upgrade supervisely[apps]==$tag_ref_name
RUN pip install --upgrade supervisely[aug]==$tag_ref_name
RUN pip install --upgrade supervisely[agent]==$tag_ref_name
