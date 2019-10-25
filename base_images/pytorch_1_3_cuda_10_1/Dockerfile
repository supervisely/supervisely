ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py-cuda-10-1:${TAG}

##############################################################################
# pytorch
##############################################################################
RUN pip install --no-cache-dir \
        torch==1.3.0 \
        torchvision==0.4.1 \
        -f https://download.pytorch.org/whl/torch_stable.html