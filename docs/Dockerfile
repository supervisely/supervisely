FROM sphinxdoc/sphinx

RUN apt-get update
RUN apt install curl -y
RUN curl -L -o panda.deb https://github.com/jgm/pandoc/releases/download/2.12/pandoc-2.12-1-amd64.deb && \
    dpkg -i panda.deb && \
    rm panda.deb

RUN pip install sphinx \
sphinx-rtd-theme \
sphinx-copybutton \
m2r2 \
nbsphinx
RUN pip install python-json-logger
RUN pip install numpy
RUN pip install pandas
RUN pip install opencv-python
RUN pip install supervisely
RUN apt-get install libgl1 -y


COPY entrypoint.sh /app/

WORKDIR /app

CMD ["bash", "/app/entrypoint.sh"]
