ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}

##############################################################################
# Additional project libraries
##############################################################################

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        lshw=02.18.85-0.3ubuntu2 \
        aha=0.5-1 \
        html2text=1.3.2a-25 \
        htop=2.2.0-2build1 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN pip install --no-cache-dir \
        docker==3.3.0 \
        psutil==5.4.5 \
        requests==2.24.0 \
        hurry.filesize==0.9 \
        scandir==1.7 \
        grpcio==1.34.1 \
        grpcio-tools==1.34.1 \
        py3exiv2==0.9.3

RUN pip install requests-toolbelt
RUN pip install packaging

RUN apt-get update \
    && apt-get install -y tree \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN pip install docker --upgrade
RUN pip install version-parser==1.0.1

############### copy code ###############
ARG MODULE_PATH
COPY $MODULE_PATH /workdir
COPY supervisely_lib /workdir/supervisely_lib

ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir/src

ENTRYPOINT ["sh", "-c", "python -u /workdir/src/main.py"]
