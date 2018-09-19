FROM supervisely/nn-base

# pytorch
RUN conda install -y -c soumith \
        magma-cuda90=2.3.0 \
    && conda install -y -c pytorch \
        pytorch=0.3.1 \
        torchvision=0.2.0 \
        cuda90=1.0 \
    && conda clean --all --yes

# sources
ENV PYTHONPATH /workdir:$PYTHONPATH
WORKDIR /workdir/src

ARG SOURCE_PATH
ARG RUN_SCRIPT
ENV RUN_SCRIPT=$RUN_SCRIPT

COPY supervisely_lib /workdir/supervisely_lib
COPY $SOURCE_PATH /workdir/src

ENTRYPOINT ["sh", "-c", "python -u ${RUN_SCRIPT}"]
