# ============================================================================
# STAGE 1: Builder Stage
# ============================================================================
ARG tag_ref_name=6.73.531

FROM supervisely/base-py-sdk-light:$tag_ref_name AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  libaec-dev \
  libgeos-c1v5 \
  && rm -rf /var/lib/apt/lists/*

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

# todo: remove; test only
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images
RUN pip install protobuf==6.33.5
RUN pip install pillow==12.1.1
RUN pip install fastapi==0.129.0
RUN pip install starlette==0.49.1

# ============================================================================
# STAGE 2: Runtime Stage (Minimal & Hardened)
# ============================================================================
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1

# Copy ONLY required system libraries from builder
COPY --from=builder /usr/lib/x86_64-linux-gnu/libSM.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libXext.so.6 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgeos_c.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libavcodec.so.* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libmagic.so.1 /usr/lib/x86_64-linux-gnu/

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libc6 \
    libc-bin \
    libglib2.0-0 \
    libgl1 \
    libsqlite3-0 \
    zlib1g \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip==26.0.1