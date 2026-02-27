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
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1
RUN pip install setuptools==78.1.1

RUN pip install dropbox==11.36.2

# RUN pip install --upgrade supervisely==$tag_ref_name
# todo: remove; test only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images

RUN pip install --upgrade \
  "protobuf==6.33.5" \
  "jaraco.context==6.1.0" \
  wheel==0.46.2 \
  && rm -rf \
  /opt/venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.context* \
  /opt/venv/lib/python3.10/site-packages/setuptools/_vendor/wheel*

# ============================================================================
# STAGE 2: Runtime Stage (Hardened)
# ============================================================================
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

# Copy Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN /usr/local/bin/strip-venv-packagers