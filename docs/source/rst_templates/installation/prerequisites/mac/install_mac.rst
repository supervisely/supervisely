Prerequisites for MacOS
=======================

All you need is Python and Homebrew installed, in order to install supervisely properly.

.. note::
   If you already have Python and Homebrew installed, you can skip step 1 & 3. Recommended version: Python 3.8

Option 1: Pure Python
---------------------

1. Install package manager `Homebrew <https://brew.sh/>`_.

.. code-block:: bash

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    brew update


2. Install `git` and `md5sum` utilities. They will be needed later.

.. code-block:: bash

    brew install git
    brew install md5sha1sum


3. Install **Python 3.8**.

.. code-block:: bash



Option 2: Miniconda
-------------------

1. Install package manager `Homebrew <https://brew.sh/>`_.

.. code-block:: bash

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    brew update


2. Install `git`, `md5sum` and other utilities. They will be needed later.

.. code-block:: bash

    brew install git
    brew install md5sha1sum


3. Install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code-block:: bash

    mkdir -p ~/temp
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-MacOSX-x86_64.sh -O ~/temp/miniconda.sh
    sudo bash ~/temp/miniconda.sh -bfp /usr/local
    sudo conda install -y python=3.8
    sudo conda clean --all --yes
    rm -rf ~/temp/miniconda.sh
