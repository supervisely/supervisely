Prerequisites for Linux
=======================

All you need is Python and pip installed, in order to install supervisely properly.

.. note::
   If you already have Python and pip installed, you can skip step 3. Recommended version: Python 3.8


Option 1: Pure Python
---------------------

1. Update package manager.

.. code-block:: bash

    apt-get update


2. Install `libjpeg-dev`, `libpng-dev`, `software-properties-common` and `ffmpeg` utilities.

.. code-block:: bash

    apt-get install -y --no-install-recommends libjpeg-dev libpng-dev software-properties-common ffmpeg
    add-apt-repository universe

3. Install **Python**.

.. code-block:: bash

    apt-get install -y python3.8 python3-pip
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

Option 2: Miniconda
-------------------

1. Update package manager.

.. code-block:: bash

    apt-get update


2. Install `git` and other utilities. They will be needed later.

.. code-block:: bash

    apt-get install -y git curl ffmpeg


3. Install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code-block:: bash

    mkdir -p ~/temp
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -o ~/temp/miniconda.sh
    sudo bash ~/temp/miniconda.sh -bfp /usr/local
    conda install -y python=3.8
    conda clean --all --yes
    rm -rf ~/temp/miniconda.sh
