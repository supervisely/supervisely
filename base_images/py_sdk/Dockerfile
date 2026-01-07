# https://hub.docker.com/r/supervisely/base-py-sdk
FROM supervisely/base-py:6.1.8

RUN apt-get update
RUN apt-get install -y zlib1g


RUN pip install --upgrade pip
RUN pip install attrs
RUN pip install requests==2.28.0
RUN apt-get install -y libmagic-dev
RUN pip install Werkzeug==2.2.1
RUN pip install MarkupSafe==2.1.1
RUN pip install beautifulsoup4==4.12.3
RUN pip install ruamel.yaml==0.17.21
# to install all SDK dependencies


ARG tag_ref_name

RUN pip install --upgrade supervisely==$tag_ref_name
RUN pip install --upgrade supervisely[apps]==$tag_ref_name
RUN pip install --upgrade supervisely[aug]==$tag_ref_name
RUN pip install --upgrade supervisely[agent]==$tag_ref_name
RUN pip install --upgrade supervisely[versioning]==$tag_ref_name
