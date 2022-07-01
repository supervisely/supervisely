Installation
============

Installation from pip
---------------------

To install Supervisely from pip simply type the following command in terminal:

.. code-block:: bash

    pip install supervisely

The above command installs only recommended dependencies, that are required for general usage. If
you face some import errors, then you should install extra dependencies (used only in limited scenarios). To install
extras dependencies use the following command:

.. code-block:: bash

    pip install 'supervisely[extras]'

We release updates quite often, so use following command if you would like to upgrade your current Supervisely package:

.. code-block:: bash

    pip install supervisely --upgrade

.. note::
   The only prerequisites are Python >= 3.8 and pip.

.. tip:: opencv-python may require:

   .. code-block:: bash

         apt-get install libgtk2.0-dev

   You can also use pre-built Docker image which can be found on Docker Hub:

   .. code-block:: bash

      docker pull supervisely/base-py

   The corresponding Dockerfile can be found in base_images directory.

Installation from Source
------------------------

Clone the repository and create a linked install.
This will allow you to change files in the
supervisely directory, and is great
if you want to modify the Supervisely library code.

.. code-block:: bash

    git clone https://github.com/supervisely/supervisely.git && \
    pip install -e ./supervisely

Or

.. code-block:: bash

    python -m pip install git+https://github.com/supervisely/supervisely.git



