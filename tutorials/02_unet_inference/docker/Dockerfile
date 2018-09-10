FROM supervisely/nn-unet-v2-base


##############################################################################
# common
##############################################################################

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.1ubuntu2 \
        curl=7.47.0-1ubuntu2.8 \
        ca-certificates=20170717~16.04.1 \
        libjpeg-dev=8c-2ubuntu8 \
        libpng-dev \
        software-properties-common=0.96.20.7 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log


##############################################################################
# Miniconda & python 3.6
##############################################################################
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6.5 \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH


##############################################################################
# sly dependencies
##############################################################################

# opencv; other packages are deps & mentioned to fix versions
RUN conda install -y -c menpo \
        opencv=3.4.1 \
        numpy=1.14.3 \
    && conda clean --all --yes


##############################################################################
# Jupyter & matplotlib
##############################################################################
RUN pip install jupyter \
        matplotlib
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
RUN pip install shapely python-json-logger scikit-image
ENV PYTHONPATH /src/unet_src
WORKDIR /src

