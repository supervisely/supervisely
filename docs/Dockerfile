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
RUN apt-get install libgl1 -y
RUN pip install supervisely

RUN pip install furo

COPY entrypoint.sh /app/

WORKDIR /app

CMD ["bash", "/app/entrypoint.sh"]
