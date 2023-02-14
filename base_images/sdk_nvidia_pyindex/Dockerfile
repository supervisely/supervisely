FROM supervisely/base-py-sdk:6.66.4

# supervisely/sdk-nvidia-pyindex:<sdk version>

ARG tag_ref_name

RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorflow
RUN pip install nvidia-tensorflow[horovod]
RUN pip install ruamel.yaml==0.17.21

RUN pip install --upgrade supervisely==$tag_ref_name
RUN pip install --upgrade supervisely[apps]==$tag_ref_name
RUN pip install --upgrade supervisely[aug]==${tag_ref_name}
