FROM supervisely/base-py:6.1.7

ARG tag_ref_name

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get install -y zlib1g


RUN pip install attrs
RUN pip install --upgrade pip
RUN pip install requests==2.28.0
RUN apt-get install -y libmagic-dev
RUN pip install Werkzeug==2.2.1
RUN pip install MarkupSafe==2.1.1
RUN pip install beautifulsoup4
# to install all SDK dependencies
RUN pip install supervisely==6.66.0

RUN pip install --upgrade supervisely==$tag_ref_name
RUN pip install --upgrade supervisely[apps]==$tag_ref_name
