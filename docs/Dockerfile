FROM docker.deepsystems.io/supervisely/five/base-py:6

RUN curl -L -o panda.deb https://github.com/jgm/pandoc/releases/download/2.12/pandoc-2.12-1-amd64.deb && \
    dpkg -i panda.deb && \
    rm panda.deb

RUN pip install sphinx==3.4.3 \
sphinx-rtd-theme==0.5.1 \
sphinx-copybutton==0.3.1 \
m2r2==0.2.7 \
nbsphinx==0.8.1 \
# sly reqs \
torch==1.7.1 \
torchvision==0.8.2

COPY entrypoint.sh /app/

WORKDIR /app

CMD ["bash", "/app/entrypoint.sh"]