ARG tag_ref_name

# ============================================================================
# STAGE 1: CUDA Stage
# ============================================================================
FROM python:3.14-slim-bookworm AS cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ca-certificates \
		wget \
	&& rm -rf /var/lib/apt/lists/*

RUN set -eux; \
	wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; \
	dpkg -i /tmp/cuda-keyring.deb; \
	rm -f /tmp/cuda-keyring.deb; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		cuda-toolkit-12-8 \
		cuda-cudart-12-8 \
		libcublas-12-8 \
		libcufft-12-8 \
		libcurand-12-8 \
		libcusolver-12-8 \
		libcusparse-12-8; \
	rm -rf /var/lib/apt/lists/*

# ============================================================================
# STAGE 2: Builder Stage (Python deps)
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
		ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install pip==26.0.1 setuptools==80.10.1 wheel==0.46.2
RUN pip install pip-audit==2.7.3

ARG tag_ref_name

# Install supervisely package with retry logic for PyPI CDN propagation
# RUN sh -c 'set -eu && \
#  MAX_ATTEMPTS=8 && \
#  TIMEOUT=15 && \
#  ATTEMPT=0 && \
#  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do \
#  echo "Attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS: Installing supervisely==$tag_ref_name..." && \
#  if pip install --no-cache-dir --upgrade supervisely==$tag_ref_name; then \
#  echo "Supervisely installed successfully" && \
#  exit 0; \
#  fi && \
#  ATTEMPT=$((ATTEMPT + 1)) && \
#  if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then \
#  echo "Installation failed. Waiting ${TIMEOUT}s..." && \
#  sleep $TIMEOUT; \
#  fi; \
#  done && \
#  echo "ERROR: Failed to install supervisely==$tag_ref_name after $MAX_ATTEMPTS attempts" >&2 && \
#  exit 1'

# todo: remove; tests only -------------------------------------
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		git \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/supervisely/supervisely.git@hardened-docker-images
RUN pip install pillow==12.1.1 wheel==0.46.2
# ---------------------------------------------------------------

RUN rm -rf \
	/opt/venv/lib/python3.14/site-packages/pip* \
	/opt/venv/lib/python3.14/site-packages/wheel* \
	/opt/venv/lib/python3.14/site-packages/setuptools/_vendor/wheel* \
	/opt/venv/bin/pip \
	/opt/venv/bin/pip3 \
	/opt/venv/bin/pip3.* \
	/opt/venv/bin/easy_install \
	/opt/venv/bin/easy_install-* \
	/opt/venv/bin/wheel

# ============================================================================
# STAGE 3: Runtime Stage (Hardened + CUDA)
# ============================================================================
FROM supervisely/py-sdk-hardened:${tag_ref_name} AS runtime

COPY --from=builder /opt/venv /opt/venv

COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufft.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusolver.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusparse.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvJitLink.so* /usr/lib/

ENV PATH="/opt/venv/bin:$PATH" \
	LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

ENV NVIDIA_VISIBLE_DEVICES=all \
	NVIDIA_DRIVER_CAPABILITIES=compute,utility
