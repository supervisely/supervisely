ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py-cuda-10-1:${TAG}

##############################################################################
# pytorch
##############################################################################
RUN conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch

RUN pip install "pillow<7"