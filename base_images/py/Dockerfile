#FROM supervisely/base-py:6.0.17
#FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04

FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

##############################################################################
# common
##############################################################################
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
# Miniconda & python 3.9
##############################################################################
# RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -o /tmp/miniconda.sh \
#     && bash /tmp/miniconda.sh -bfp /usr/local \
#     && rm -rf /tmp/miniconda.sh \
#     && conda install -y python=3.8

#ENV PATH /opt/conda/bin:$PATH

RUN add-apt-repository universe
RUN apt-get update
RUN apt-get install -y python3.8 python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

##############################################################################
# sly dependencies
##############################################################################
# libgeos for shapely; other are deps of cv2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgeos-dev=3.8.0-1build1 \
        libsm6=2:1.2.3-1 \
        libxext6=2:1.3.4-0ubuntu1 \
        libxrender-dev=1:0.9.10-1 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN pip install \
        python-json-logger==0.1.11 \
        pybase64==1.0.2 \
        shapely==1.7.1 \
        imgaug==0.4.0 \
        numpy==1.19 \
        opencv-python==3.4.10.35 \
        scipy==1.5.2 \
        scikit-image==0.17.1 \
        matplotlib==3.3.2 \
        pillow==5.4.1 \
        requests==2.24.0 \
        networkx==2.5 \
        jsonschema==2.6.0

##############################################################################
# Java to run Pycharm
##############################################################################
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        default-jre=2:1.11-72 \
        default-jdk=2:1.11-72 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && rm -rf /jre \
    && ln -s /usr/lib/jvm/java-7-openjdk-amd64 /jre

ENV JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64


##############################################################################
# Additional project libraries
##############################################################################
RUN pip install \
    pandas==1.1.3 \
    grpcio==1.32.0 \
    grpcio-tools==1.32.0

RUN pip install --upgrade pip
RUN apt-get update && \
    apt-get -y install \
        libexiv2-dev \
        libboost-all-dev

#RUN pip install py3exiv2==0.8.0
#RUN pip install simplejson==3.17.2
RUN pip install requests-toolbelt==0.9.1
RUN pip install PTable
RUN pip install flask-restful
RUN apt-get install -y fonts-noto=20200323-1build1~ubuntu20.04.1

RUN pip install pascal-voc-writer

RUN pip install bidict==0.21.2
RUN pip install scikit-video==1.1.11

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd

RUN pip install plotly==4.11.0
RUN pip install docker==4.4.1
RUN pip install fuzzywuzzy

# to support pandas.read_excel
RUN pip install xlrd
RUN pip install google-cloud-storage
RUN pip install python-slugify
RUN pip install psutil==5.4.5
RUN pip install cython

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install protobuf==3.14.0
RUN pip install Werkzeug==1.0.1
RUN pip install sk-video==1.1.10

RUN /usr/bin/python3 -m pip install --upgrade pip
##############################################################################
# Encoding for python SDK
##############################################################################
ENV LANG C.UTF-8
