ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}

RUN pip install attrs

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN python -m pip install git+https://github.com/supervisely/supervisely.git
