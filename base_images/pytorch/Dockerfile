ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}

##############################################################################
# pytorch
##############################################################################
RUN conda install -y -c soumith \
        magma-cuda90=2.3.0 \
    && conda install -y -c pytorch \
        pytorch=0.3.1 \
        torchvision=0.2.0 \
        cuda90=1.0 \
    && conda clean --all --yes


