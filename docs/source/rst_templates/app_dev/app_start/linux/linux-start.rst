Prerequisites for Linux
=======================

.. note::
   Install your favorite Python IDE. We prefer and use PyCharm. The further steps will be shown using PyCharm CE (Community Edition).

1. Install PyCharm
------------------

Install `Pycharm <https://www.jetbrains.com/pycharm/download/#section=linux>`_ Community or Professional Edition.
You can do it by clicking on the Download button.

.. image:: images/install-linux.png
   :scale: 50%

1. Unpack the PyCharm distribution archive that you downloaded
   where you wish to install the program. We will refer to this
   location as your {installation home}.

2. To start the application, open a console, cd into "{installation home}/bin" and type:

.. code-block:: bash

    ./pycharm.sh

You can also install Pycharm with command prompt:

For **Pycharm Community Edition**:

.. code-block:: bash

    sudo snap install pycharm-community --classic

For **Pycharm Professional Edition**:

.. code-block:: bash

    sudo snap install pycharm-professional --classic

2. Configure PyCharm
--------------------

Open Pycharm and install those Plugins:

     * .env files support
     * EnvFile
     * Requirements

.. image:: images/install-plugins.png
   :scale: 50%

3. Prepare Working Directory
----------------------------

1. Download our Application `template <https://github.com/supervisely-ecosystem/app-template-headless/archive/refs/heads/master.zip>`_
from `GitHub <https://github.com/supervisely-ecosystem/app-template-headless>`_.
Unpack it to your working directory (e.g: /home/admin/work/app-dev).

2. Open downloaded project in PyCharm

.. image:: images/open-project-1.png
   :scale: 50%

.. image:: images/open-project-2.png
   :scale: 50%

4. Configure PyCharm
--------------------

1. Add Pycharm Python Interpreter: File -> Settings -> Project -> Python Interpreter -> Press on Gear icon -> Add.

.. image:: images/settings-interpreter.png
   :scale: 50%

2. Configure Python Interpreter.

.. image:: images/add-intepreter.png
   :scale: 50%

3. Select Python Interpreter

.. image:: images/select-interpreter.png
   :scale: 50%

4. Directory **venv** should appear in your project now.

.. image:: images/venv-appear.png
   :scale: 50%

4. Configure PyCharm
--------------------

1. Open **requirements.txt** right click in IDE and select **Install All Packages**.

.. image:: images/install-reqs.png
   :scale: 50%

2. Press ``Add configuration`` -> **Edit configurations** in top right corner.

.. image:: images/add-conf.png
   :scale: 50%

3. Add **+** Python and setup new configuration. And add **.env** files to it.
**secret_debug.env** contains your personal Supervisely credentials and must overwrite **debug.env**.

.. image:: images/setup-conf.png
   :scale: 50%

.. image:: images/add-conf-env.png
   :scale: 50%

5. Final Steps
--------------

1. Last steps are all about setting up environment files.
Add following lines to your **debug.env**:

.. code-block:: python

   DEBUG_APP_DIR="/path/to/app_debug_data"
   DEBUG_CACHE_DIR="/path/to/app_debug_cache"
   LOG_LEVEL="debug"

2. Go to **Supervisely** -> **Ecosystem** and add **While True Script** Application.

.. image:: images/add-app-1.png
   :scale: 50%

.. image:: images/add-app-2.png
   :scale: 50%

3. Run **While True Script** Application get it's ID and insert it to **debug.env** file.

.. image:: images/app-work-id.png
   :scale: 50%

4. This is how your **debug.env** should look like after all manipulations.

.. image:: images/debug-final.png
   :scale: 80%

Configure **secret_debug.env**
------------------------------
Add Server Address variable:

.. code-block:: python

    SERVER_ADDRESS="https://app.supervise.ly/"

6. Get your Supervisely API Token and paste it to **API_TOKEN** variable.

.. image:: images/API-token.png
   :scale: 50%

.. code-block:: python

    API_TOKEN="Insert your API Token here"

7. Get your Supervisely Agent Token and paste it to **AGENT_TOKEN** variable.

.. image:: images/Agent-token.png
   :scale: 50%

.. code-block:: python

    AGENT_TOKEN="Insert your Agent Token here"

.. warning::
   **Double check that secret_debug.env is added to .gitignore!**

All Done!
---------
