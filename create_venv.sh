#!/bin/bash

# learn more in documentation
# Official python docs: https://docs.python.org/3/library/venv.html
# Superviely developer portal: https://developer.supervise.ly/getting-started/installation#venv

if [ -d ".venv" ]; then
    echo "VENV already exists, will be removed"
    rm -rf .venv
fi

echo "VENV will be created" && \
python3 -m venv .venv && \
source .venv/bin/activate && \

echo "Install requirements..." && \
pip3 install . && \
echo "Requirements have been successfully installed" && \
echo "Testing imports, please wait a minute ..." && \
python3 -c "import supervisely as sly" && \
echo "Success!" && \
deactivate