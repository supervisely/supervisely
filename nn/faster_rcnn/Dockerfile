FROM supervisely/nn-base

# sources
ENV PYTHONPATH /workdir:$PYTHONPATH
WORKDIR /workdir/src

ARG SOURCE_PATH
ARG RUN_SCRIPT
ENV RUN_SCRIPT=$RUN_SCRIPT

COPY supervisely_lib /workdir/supervisely_lib
COPY $SOURCE_PATH /workdir/src

ENTRYPOINT ["sh", "-c", "python -u ${RUN_SCRIPT}"]
