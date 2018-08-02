FROM docker.deepsystems.io/supervisely/nn/tf_ssd/base:v1.2

############### copy code ###############
ARG MODULE_PATH
ENV MODULE_PATH=$MODULE_PATH
ENV PUBLIC_PATH supervisely_lib

COPY ${MODULE_PATH}/src /workdir/src
COPY ${PUBLIC_PATH} /workdir/supervisely_lib

WORKDIR /workdir/src
ENV PYTHONPATH /workdir:/workdir/supervisely_lib/worker_proto:/workdir/src/models/research:/workdir/src/models/research/object_detection:/workdir/src/models/research/slim:$PYTHONPATH

ARG WORKER_PATH
ENV WORKER_PATH=$WORKER_PATH
RUN echo 'worker path = '${WORKER_PATH}

ENTRYPOINT ["sh", "-c", "python -u ${WORKER_PATH}"]
