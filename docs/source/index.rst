.. Supervisely Library documentation master file, created by
   sphinx-quickstart on Tue Jan 26 13:33:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Supervisely SDK for Python
==========================

Welcome to the Supervisely SDK for Python documentation for `Supervisely <https://www.supervise.ly>`_, the leading platform for entire computer vision lifecycle.
This SDK aims to make it as easy as possible to develop new apps and plugins for the Supervisely platform.
The SDK is a product of our experience developing new features for Supervisely
and contains functionality that we have found helpful and frequently needed for
python development.

**Feel free to ask questions in** `Slack <https://supervisely.slack.com/>`_.

.. raw:: html

   <tr>
     <td>
       <div style="display:inline-block;">
         <img style="width:78px;" src="_static/images/sections/bubbles/Prerequisites.png"><br/>
       </div>
       <div style="display:inline-block;">
         <h2 style="width:auto;">Prerequisites & Installation</h2>
       </div>
       <ul>
       <li>
         <a href="rst_templates/installation/prerequisites/prerequisites.html">Prerequisites</a>
         <ul>
           <li><a href="rst_templates/installation/prerequisites/linux/install_linux.html">Prerequisites for Linux</a></li>
           <li><a href="rst_templates/installation/prerequisites/mac/install_mac.html">Prerequisites for Mac</a></li>
           <li><a href="rst_templates/installation/prerequisites/windows/install_windows.html">Prerequisites for Windows</a></li>
         </ul>
       </li>
       <li>
         <a href="rst_templates/installation/installation/installation.html">Installation</a>
       </li>
       </ul>
     </td>
   </tr>

   <tr>
     <td>
       <div style="display:inline-block;">
         <img style="width:78px;" src="_static/images/sections/bubbles/Learn.png">
       </div>
       <div style="display:inline-block;">
         <h2>Tutorials</h2>
       </div>
       <ul>
         <li><a href="https://www.youtube.com/c/Supervisely/videos">Videos on YouTube</a></li>
         <li><a href="https://docs.supervise.ly/">UI Documentation</a></li>
         <li><a href="https://github.com/supervisely/supervisely/tree/master/agent">What is Supervisely Agent</a></li>
         <li><a href="https://github.com/supervisely/supervisely/blob/master/help/tutorials/06_exam_report_explanation/06_exam_report_explanation.md">Exam Reports Explained</a></li>
        </ul>
     </td>
   </tr>

   <tr>
     <td>
       <div style="display:inline-block;">
         <img style="width:78px;" src="_static/images/sections/bubbles/API.png"><br/>
       </div>
       <div style="display:inline-block;">
         <h2>API</h2>
       </div>
       <ul>
       <li><a href="sdk_packages.html">Python SDK API Reference</a></li>
       <li><a href="https://api.docs.supervise.ly/">Public REST API Reference</a></li>
      </ul>
     </td>
   </tr>

   <tr>
     <td>
       <div style="display:inline-block;">
         <img style="width:78px;" src="_static/images/sections/bubbles/Basics.png">
       </div>
       <div style="display:inline-block;">
         <h2>SDK Basics with IPython Notebooks</h2>
       </div>
       <ul>
         <li><a href="rst_templates/notebooks/ipynb/01_project_structure.html">1. Project Structure</a></li>
         <li><a href="rst_templates/notebooks/ipynb/02_data_management.html">2. Data Management</a></li>
         <li><a href="rst_templates/notebooks/ipynb/03_augmentations.html">3. Data Augmentations</a></li>
         <li><a href="rst_templates/notebooks/ipynb/04_neural_network_inference.html">4. NN: Deploy and Inference on Supervisely via API</a></li>
         <li><a href="rst_templates/notebooks/ipynb/05_neural_network_workflow.html">5. NN: Automate Training and Inference via API</a></li>
         <li><a href="rst_templates/notebooks/ipynb/06_inference_modes.html">6. Inference Modes: Full Image / Sliding Window / ROI / Bboxes</a></li>
         <li><a href="rst_templates/notebooks/ipynb/07_data_manipulation.html">7. Data Manipulation via API: Copy / Move / Delete</a></li>
         <li><a href="rst_templates/notebooks/ipynb/08_users_labeling_jobs_api.html">8. Users and Labeling Jobs API</a></li>
         <li><a href="rst_templates/notebooks/ipynb/09_1_nns_pipeline.html">9.1. Custom NN Detection and Segmentation Pipeline</a></li>
         <li><a href="rst_templates/notebooks/ipynb/09_2_nns_pipeline.html">9.2. Custom NN Multi GPU Detection and Segmentation Pipeline</a></li>
         <li><a href="rst_templates/notebooks/ipynb/10_upload_new_images.html">10. Upload Images via API</a></li>
         <li><a href="rst_templates/notebooks/ipynb/11_custom_data_pipeline.html">11. Custom Data Pipeline</a></li>
         <li><a href="rst_templates/notebooks/ipynb/12_filter_and_combine_images.html">12. Filter and Combine Images</a></li>
         <li><a href="rst_templates/notebooks/ipynb/13_nn_inference_from_sources.html">13. NN Inference from Sources</a></li>
         <li><a href="rst_templates/notebooks/ipynb/14_pixelwise_confidences.html">14. How to Work with NN Pixelwise Probabilities</a></li>
         <li><a href="https://github.com/supervisely/supervisely/tree/master/help#cookbooks">Additional Examples in our GitHub</a></li>
        </ul>
     </td>
   </tr>

   <tr>
     <td>
       <div style="display:inline-block;">
         <img style="width:78px;" src="_static/images/sections/bubbles/Code.png"><br/>
       </div>
       <div style="display:inline-block;">
         <h2>Applications Development (In Progress)</h2>
       </div>
       <ul>
       <li><a href="rst_templates/app_dev/app_start/app_start.html">Getting Started</a></li>
       <li><a href="rst_templates/app_dev/app_first/app_first.html">First Application (Coming soon)</a></li>
      </ul>
     </td>
   </tr>

   <tr>
     <td>
       <div style="display:inline-block;">
         <img style="width:78px;" src="_static/images/sections/bubbles/Repository.png">
       </div>
       <div style="display:inline-block;">
         <h2>Develop Plugins (Deprecated)</h2>
       </div>
       <ul>
         <li><a href="https://github.com/supervisely/supervisely/blob/master/help/tutorials/01_create_new_plugin/how_to_create_plugin.md">Custom Plugin Basics</a></li>
         <li><a href="https://github.com/supervisely/supervisely/blob/master/help/tutorials/03_custom_neural_net_plugin/custom_nn_plugin.md">Custom NN Plugin Basics</a></li>
         <li><a href="https://github.com/supervisely/supervisely/blob/master/help/tutorials/05_develop_nn_plugin/develop_plugin.md">How to Debug Custom NN Plugin</a></li>
         <li><a href="https://github.com/supervisely/supervisely/blob/master/help/tutorials/04_deploy_neural_net_as_api/deploy-model.md">Different Ways How to Deploy NN with Supervisely</a></li>
        </ul>
     </td>
   </tr>

   <tr>
     <td>
       <div style="display:inline-block;">
         <img style="width:78px;" src="_static/images/sections/bubbles/GitHub.png">
       </div>
       <div style="display:inline-block;">
         <h2>Source Code</h2>
       </div>
       <ul>
         <li><a href="https://github.com/supervisely/supervisely">Supervisely GitHub</a></li>
         <li><a href="https://github.com/supervisely-ecosystem/">Supervisely Ecosystem</a></li>
       </ul>
     </td>
   </tr>

.. toctree::
   :caption: Prerequisites & Installation
   :hidden:

   Prerequisites                <rst_templates/installation/prerequisites/prerequisites>
   Installation                 <rst_templates/installation/installation/installation>

.. toctree::
   :caption: Tutorials
   :hidden:

   Videos on YouTube            <https://www.youtube.com/c/Supervisely/videos>
   UI Documentation             <https://docs.supervise.ly/>
   What is Supervisely Agent    <https://github.com/supervisely/supervisely/tree/master/agent>
   Exam Reports Explained       <https://github.com/supervisely/supervisely/blob/master/help/tutorials/06_exam_report_explanation/06_exam_report_explanation.md>


.. toctree::
   :caption: API
   :hidden:

   sdk_packages
   Public REST API              <https://api.docs.supervise.ly/>

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :caption: SDK Basics with IPython Notebooks
   :hidden:

   SDK Basics with IPython Notebooks  <rst_templates/notebooks/notebooks>


.. toctree::
   :caption: Applications Development (In Progress)
   :hidden:

   Getting Started                       <rst_templates/app_dev/app_start/app_start>
   First Application (Coming soon)       <rst_templates/app_dev/app_first/app_first>

.. toctree::
   :caption: Develop Plugins (Deprecated)
   :hidden:

   Custom Plugin Basics                                   <https://github.com/supervisely/supervisely/blob/master/help/tutorials/01_create_new_plugin/how_to_create_plugin.md>
   Custom NN Plugin Basics                                <https://github.com/supervisely/supervisely/blob/master/help/tutorials/03_custom_neural_net_plugin/custom_nn_plugin.md>
   How to Debug Custom NN Plugin                          <https://github.com/supervisely/supervisely/blob/master/help/tutorials/05_develop_nn_plugin/develop_plugin.md>
   Different Ways How to Deploy NN with Supervisely       <https://github.com/supervisely/supervisely/blob/master/help/tutorials/04_deploy_neural_net_as_api/deploy-model.md>

.. toctree::
   :caption: Source Code
   :hidden:

   Supervisely GitHub                   <https://github.com/supervisely/supervisely>
   Supervisely Ecosystem                <https://github.com/supervisely-ecosystem/>

.. toctree::
   :maxdepth: 1
   :caption: Troubleshooting
   :hidden:

   rst_templates/troubleshooting/install_issues/install_issues
