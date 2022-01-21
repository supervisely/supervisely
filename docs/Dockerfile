FROM sphinxdoc/sphinx

RUN apt-get update
RUN apt install curl -y
RUN curl -L -o panda.deb https://github.com/jgm/pandoc/releases/download/2.12/pandoc-2.12-1-amd64.deb && \
    dpkg -i panda.deb && \
    rm panda.deb

RUN pip install sphinx==4.4.0 \
sphinx-material==0.0.35 \
sphinx-copybutton==0.4.0 \
m2r2==0.3.2 \
nbsphinx==0.8.8
RUN apt-get install libgl1 -y
RUN pip install supervisely

COPY entrypoint.sh /app/

WORKDIR /app

CMD ["bash", "/app/entrypoint.sh"]
