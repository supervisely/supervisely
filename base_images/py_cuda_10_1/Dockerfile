FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

##############################################################################
# common
##############################################################################
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        nano \
        ca-certificates\
        libjpeg-dev \
        libpng-dev \
        software-properties-common \
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
# libgeos for shapely; other are deps of cv2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgeos-dev=3.5.0-1ubuntu2 \
        libsm6=2:1.2.2-1 \
        libxext6=2:1.3.3-1 \
        libxrender-dev=1:0.9.9-0ubuntu1 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN pip install --no-cache-dir \
        python-json-logger==0.1.8 \
        pybase64==0.4.0 \
        shapely==1.5.13 \
        imgaug==0.2.5 \
        numpy==1.14.3 \
        opencv-python==3.4.1.15 \
        scipy==1.1.0 \
        scikit-image==0.13.0 \
        matplotlib==2.2.2 \
        pillow==5.1.0 \
        requests==2.18.4 \
        networkx==2.1 \
        jsonschema==2.6.0


##############################################################################
# Java to run Pycharm
##############################################################################
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jre=2:1.8-56ubuntu2 \
        default-jdk=2:1.8-56ubuntu2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && ln -s /usr/lib/jvm/java-7-openjdk-amd64 /jre

ENV JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64


##############################################################################
# Additional project libraries
##############################################################################
RUN pip install --no-cache-dir \
    pandas==0.22.0 \
    grpcio==1.12.1 \
    grpcio-tools==1.12.1

RUN pip install --upgrade pip
RUN apt-get update && \
    apt-get -y install \
        libexiv2-14 \
        libexiv2-dev \
        libboost-all-dev

RUN pip install py3exiv2==0.4.0 
RUN pip install simplejson==3.16.0
RUN pip install requests-toolbelt
RUN pip install PTable
RUN pip install flask-restful
RUN apt-get install -y fonts-noto=20160116-1

RUN pip install pascal-voc-writer

RUN pip install bidict
RUN pip install --no-cache-dir scikit-video==1.1.11

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

RUN pip install plotly==4.5.4


RUN ln -sf /usr/local/bin/python3.6 /usr/bin/python

##############################################################################
# Encoding for python SDK
##############################################################################
ENV LANG C.UTF-8