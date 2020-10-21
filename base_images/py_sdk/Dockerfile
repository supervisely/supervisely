ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}

RUN pip install attrs
RUN python -m pip install git+https://github.com/supervisely/supervisely.git
