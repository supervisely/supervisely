FROM supervisely/base-py:6.1.7

ARG tag_ref_name

RUN apt-get update
RUN apt-get install -y zlib1g

RUN pip install attrs
RUN pip install --upgrade pip
RUN pip install requests==2.27.1


RUN pip install supervisely==$tag_ref_name
RUN pip install supervisely[apps]==$tag_ref_name