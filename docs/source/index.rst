.. Supervisely Library documentation master file, created by
   sphinx-quickstart on Tue Jan 26 13:33:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Supervisely SDK for Python
==========================

.. image:: _static/images/logo-dark-custom-row.png
   :width: 100%
   :alt: supervisely
   :align: center


.. raw:: html

   <p align="center">
     <a href="https://github.com/supervisely/supervisely"> <img src="https://img.shields.io/uptimerobot/status/m778791913-8b2f81d0f1c83da85158e2a5.svg"> </a>
     <a href="https://github.com/supervisely/supervisely"> <img src="https://img.shields.io/uptimerobot/ratio/m778791913-8b2f81d0f1c83da85158e2a5.svg"> </a>
     <a href="https://github.com/supervisely/supervisely"> <img src="https://img.shields.io/github/repo-size/supervisely/supervisely.svg"> </a>
     <a href="https://github.com/supervisely/supervisely"> <img src="https://img.shields.io/github/languages/top/supervisely/supervisely.svg"> </a>
     <a href="https://pypi.org/project/supervisely" target="_blank"> <img src="https://img.shields.io/pypi/v/supervisely?color=%2334D058&label=pypi%20package" alt="Package version"> </a>
     <a href="https://github.com/supervisely/supervisely"> <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"> </a>
   </p>

About Supervisely SDK
---------------------
This SDK aims to make it as easy as possible to develop new apps and plugins for the `Supervisely platform <https://www.supervise.ly>`_.
The SDK is a product of our experience developing new features for Supervisely
and contains functionality that we have found helpful and frequently needed for
python development.

**Feel free to ask questions in** `Slack <https://supervisely.slack.com/>`_.


-----------
Readme File
-----------

.. mdinclude:: ../../README.md


Quick Start
-----------

Pip
^^^

Simply type the following command in terminal to install Supervisely:

.. code-block:: bash

    pip install supervisely

.. tip:: opencv-python may require:

   .. code-block:: bash

         apt-get install libgtk2.0-dev

Docker
^^^^^^

Or use pre-built Docker image which can be found on Docker Hub:

 .. code-block:: bash

    docker pull supervisely/base-py

.. note:: The corresponding Dockerfile can be found in base_images directory in supervisely github repository.

Source
^^^^^^

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


.. toctree::
   :maxdepth: 1

   rst_templates/installation/installation
   sdk_packages
   rst_templates/tutorials/tutorials
   rst_templates/troubleshooting/troubleshooting



.. toctree::
   :caption: Source Code
   :hidden:

   Supervisely GitHub                   <https://github.com/supervisely/supervisely>
   Supervisely Ecosystem                <https://github.com/supervisely-ecosystem/>


