ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}
##############################################################################
# Additional project libraries
##############################################################################

RUN pip install --no-cache-dir \
        scikit-video==1.1.11 \
        dhash==1.3

############### copy code ###############
ARG MODULE_PATH
COPY $MODULE_PATH /workdir
COPY supervisely_lib /workdir/supervisely_lib

ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir/src