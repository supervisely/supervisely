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
  git \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1 wheel==0.46.2

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
  -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install 'protobuf>=3.19.6, <=3.20.3'
RUN pip install Cython==3.0.0 tensorboard==2.13.0
RUN pip install seaborn==0.12.2 ftfy==6.1.1 regex==2023.6.3
RUN pip install git+https://github.com/supervisely-ecosystem/depends-CLIP.git
RUN pip install imutils==0.5.4 loguru==0.7.0

# Supervisely
RUN pip install --upgrade supervisely==$tag_ref_name

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
