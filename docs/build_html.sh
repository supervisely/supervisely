#!/bin/bash  

# docs/source
# must contain conf.py and index.rst
# both files were created automatically by sphinx-quickstart command
# and manually edited

# pip install sphinx
# pip install sphinx-rtd-theme
# pip install sphinx-copybutton
# pip install m2r2
# pip install nbsphinx
# curl -L -o panda.deb https://github.com/jgm/pandoc/releases/download/2.12/pandoc-2.12-1-amd64.deb

# docl sphinx-docs

# to clean build directory use
# `make clean` command in terminal

#only for new modules and lib updates
#sphinx-apidoc -o source/ ../supervisely_lib

#docker run command
#docker-compose up sphinx-docs

#local run command
sphinx-build source/ build/