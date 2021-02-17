ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:${TAG}

##############################################################################
# pytorch
##############################################################################
#RUN conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

#RUN pip install "pillow<7"


