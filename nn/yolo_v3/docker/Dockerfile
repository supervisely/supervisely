FROM docker.deepsystems.io/supervisely/nn/yolo_v3/base:v1.00

############### copy code ###############
ARG MODULE_PATH
ENV MODULE_PATH=$MODULE_PATH
ENV PUBLIC_PATH supervisely_lib

COPY ${MODULE_PATH}/src /workdir/src
COPY ${PUBLIC_PATH} /workdir/supervisely_lib

RUN cd /workdir/src/darknet && make clean && make -j4 all

WORKDIR /workdir/src
ENV PYTHONPATH /workdir:$PYTHONPATH

ARG WORKER_PATH
ENV WORKER_PATH=$WORKER_PATH
RUN echo 'worker path = '${WORKER_PATH}

ENTRYPOINT ["sh", "-c", "python -u ${WORKER_PATH}"]

# @TODO: add step rto build darknet
