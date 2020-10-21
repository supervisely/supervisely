ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}

##############################################################################
# pytorch
##############################################################################
RUN conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch \
    && conda clean --all --yes

RUN pip install "pillow<7"
