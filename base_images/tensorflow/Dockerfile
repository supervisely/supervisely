ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-py:6.0.17

##############################################################################
# tensorflow
##############################################################################
# downgrade cudnn for tensorflow, it doesn't like cudnn 7.1 (tested with tf 1.5, 1.7)
RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        libcudnn7=7.0.5.15-1+cuda9.0 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN pip install --no-cache-dir \
        tensorflow_gpu==1.5.0

# temp solution
RUN pip install plotly==4.11.0
RUN pip install docker
RUN pip install fuzzywuzzy

# to support pandas.read_excel
RUN pip install xlrd
RUN pip install google-cloud-storage
RUN pip install python-slugify
RUN pip install psutil==5.4.5
RUN pip install cython