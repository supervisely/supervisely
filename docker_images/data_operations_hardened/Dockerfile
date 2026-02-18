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
  ca-certificates \
  libzbar0 \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1 wheel==0.46.2

RUN pip install pyzbar==0.1.9
RUN pip install moviepy==1.0.3 imageio-ffmpeg==0.4.7
RUN pip install imutils==0.5.4
RUN pip install decord==0.6.0

# RUN pip install --upgrade supervisely==$tag_ref_name
# RUN pip install --upgrade supervisely[docs]==$tag_ref_name
# todo: remove; test only
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

# Copy required system libraries from builder
COPY --from=builder /usr/lib/*-linux-gnu/libzbar.so.* /usr/lib/

# Copy Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
