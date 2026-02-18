# ============================================================================
# STAGE 1: Builder Stage
# ============================================================================
ARG tag_ref_name

FROM python:3.10-slim-bookworm AS builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  git \
  libaec-dev \
  libavcodec59 \
  libgeos-c1v5 \
  libglib2.0-0 \
  libgl1 \
  libmagic1 \
  libsm6 \
  libxext6 \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1 wheel==0.46.2

# Install runtime dependencies
RUN pip install --no-build-isolation pycocotools==2.0.6
RUN pip install htmllistparse==0.6.0
RUN pip install google-cloud-storage==1.35.0
RUN pip install google-api-python-client==2.81.0
RUN pip install moviepy==1.0.3
RUN pip install open3d==0.19.0
RUN pip install scikit-learn==1.5.0
RUN pip install laspy[lazrs]==2.5.3
RUN pip install pypcd4==1.4.3
RUN pip install git+https://github.com/autonomousvision/kitti360Scripts.git@32f6d64
RUN pip install transforms3d==0.4.2
RUN pip install gdcm==1.1
RUN pip install pylibjpeg==1.4.0
RUN pip install pylibjpeg-libjpeg==2.1.0
RUN pip install pillow-avif-plugin==1.4.3
RUN pip install pillow-heif==0.18.0
RUN pip install tifffile==2023.7.10
# RUN pip install pymupdf==1.22.5
RUN pip install nibabel==5.2.1
# RUN pip install bagpy==0.5
RUN pip install imagecodecs==2024.6.1
RUN pip install lyft-dataset-sdk==0.0.8
RUN pip install matplotlib==3.8.4
RUN pip install shapely==2.0.4
RUN pip install nuscenes-devkit==1.1.11 --no-deps
RUN pip install lxml==5.2.2
RUN pip install scikit-image==0.23.2

# RUN pip install --upgrade supervisely==$tag_ref_name
# todo: remove; test only
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

# Copy Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"