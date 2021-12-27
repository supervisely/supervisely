<p align="center">
  <a href="https://supervise.ly"><img alt="Supervisely" title="Supervisely" src="https://i.imgur.com/B276eMS.png" width=450></a>
</p>

<h3 align="center">
      The leading platform for entire computer vision lifecycle
    </h3>

<div class="subtitle">
  <p align="center">Iterate from image annotation to accurate Neural Networks 10x faster</p>
    </div>


---------------------------------------------------------------------------

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue">
  <img src="https://img.shields.io/pypi/v/supervisely?color=brightgreen&label=pypi%20package">
  <img src="https://img.shields.io/uptimerobot/status/m778791913-8b2f81d0f1c83da85158e2a5.svg">
  <img src="https://img.shields.io/github/repo-size/supervisely/supervisely.svg">
  <a href="https://supervisely.slack.com/"> <img src="https://img.shields.io/badge/slack-@supervisely-yellow.svg?logo=slack"></a>
</p>


- [About Supervisely](#about-supervisely)
- [Documentation](#documentation)
- [Agent](#agent)
- [Supervisely Ecosystem](#supervisely-ecosystem)
- [Python SDK](#python-sdk)
- [Neural Networks](#neural-networks)
- [Import plugins](#import-plugins)
- [Data Transformation Language](#data-transformation-language)
- [Resourses](#resources)
- [The Team](#the-team)

# <img src="https://i.imgur.com/8hnsGt5.png"> About Supervisely
Supervisely is a web platform where you can find everything you need to build Deep Learning solutions within a single environment.
You can think of Supervisely as an Operating System available via Web Browser to help you solve Computer Vision tasks. The idea is to unify all the relevant tools that may be needed to make the development process as smooth and fast as possible.

More concretely, Supervisely includes the following functionality:

* Data labeling for images, videos, 3D point cloud and volumetric medical images (dicom)
* Data visualization and quality control
* State-Of-The-Art Deep Learning models for segmentation, detection, classification and other tasks
* Interactive tools for model performance analysis
* Specialized Deep Learning models to speed up data labeling (aka AI-assisted labeling)
* Synthetic data generation tools
* Instruments to make it easier to collaborate for data scientists, data labelers, domain experts and software engineers

One challenge is to make it possible for everyone to train and apply SOTA Deep Learning models directly from the Web Browser. To address it, we introduce an open sourced Supervisely Agent. All you need to do is to execute a single command on your machine with the GPU that installs the Agent. After that, you keep working in the browser and all the GPU related computations will be performed on the connected machine(s).

We learn a lot from our awesome community and want to give something back.
Here you can find the Python SDK we use to implement neural network models,
import plugins for custom data formats, and tools like the Data Transformation
Language. You can also find the source code for the agent to turn your PC into a
worker node to deploy your neural nets.

# <img src="https://i.imgur.com/2PpeTza.png"> Documentation

* [Supervisely UI Documentation](https://docs.supervise.ly/)
* [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi)
* [Supervisely REST API](https://api.docs.supervise.ly/)
* [Supervisely SDK for Python](https://sdk.docs.supervise.ly/)

# <img src="https://i.imgur.com/GS2q7BK.png"> Agent

![Docker Pulls](https://img.shields.io/docker/pulls/supervisely/agent?color=informational&logo=docker)

Supervisely Agent is a simple open-source task manager for Supervisely available as a Docker image. This is the simple way how you can connect your computational resources: your private server or cloud on (AWS, Google Cloud, Azure and so on).
Run command in Terminal on your computer to connect Agent to the Supervisely WEB instance. It listens for the tasks (like neural net training request), handles downloading and uploading data to the web instance, sets up proper environments for the specific tasks, and keeps track of progress, successes and failures of individual tasks.

You should connect computer with GPU to your Supervisely account. If you already have Supervisely Agent running on your computer, you can skip this step.
Several tools have to be installed on your computer:

* Nvidia drives + CUDA Toolkit
* Docker
* NVIDIA Container Toolkit
* 
Once your computer is ready just add agent to your team and execute automatically generated running command in terminal. Watch how-to video:

[![Watch the video](https://img.youtube.com/vi/aDqQiYycqyk/0.jpg)](https://www.youtube.com/watch?v=aDqQiYycqyk)

Check out [explanation](https://github.com/supervisely/supervisely/tree/master/agent) on how agent works and [documentation](https://docs.supervise.ly/customization/agents) on how to deploy a new agent.

# <img src="https://i.imgur.com/SVoaRou.png"> Supervisely Ecosystem 

<a href ="https://github.com/supervisely-ecosystem"><img src="https://img.shields.io/badge/GitHub-Supervisely%20Ecosystem-blue"></a>

We designed Supervisely to be modular: importing and exporting data, neural networks training, processing with python notebooks and many more — all can be done by separate modules.
[Here](https://github.com/supervisely-ecosystem/repository) you can find a constatly growing and updating list of all publicly avaliable ecosystem applications and plugins.
If you have any ideas or request feel free to share it at [Ideas Exchange](https://ideas.supervise.ly/)

# <img src="https://i.imgur.com/9i4z8em.png"> Python SDK

<a href ="https://sdk.docs.supervise.ly/"><img src="https://img.shields.io/badge/Python%20SDK%20Documentation-Supervisely-yellow"></a>

We have organized most of the common functionality for processing data in
[Supervisely format](https://docs.supervise.ly/ann_format/) and for training and
inference with neural networks into the [Python SDK](./supervisely_lib). Our stock plugins rely on the SDK
extensively, and we believe the SDK will be also valuable to the community.

The SDK not only wraps all the low-level details of handling the data and
communicating with the Supervisely web instance, but also provides convenience
helpers for the common tasks that we found useful in our own work of developing
Supervisely plugins, such as neural network architectures and custom dataset
imports.

Key features:

 * Read, modify and write Supervisely projects on disk.
 * Work with labeling data: geometric objects and tags.
 * Common functionality for developing Supervisely plugins, so that you only
   need to focus on the core of your custom logic, and not on low level
   interactions with the platform.
   
#### Installation:

```
pip install supervisely
```

or
```
git clone https://github.com/supervisely/supervisely.git && \
pip install -e ./supervisely
```

or 
```
python -m pip install git+https://github.com/supervisely/supervisely.git
```

We release updates quite often, so use following command if you would like to upgrade you current supervisely package:
```
pip install supervisely --upgrade
```

The only prerequisites are `Python` >= 3.8 and `pip`.

Tip: `opencv-python` may require `apt-get install libgtk2.0-dev` Or use pre-built Docker image which can be found on Docker Hub:

```docker pull supervisely/base-py```

The corresponding `Dockerfile` can be found in `base_images` directory. 

# <img src="https://i.imgur.com/GeRIZSG.png"> Neural Networks

We have ported and implemented a number of popular neural network (NN)
architectures to be available as Supervisely plugins. Each plugin is a separate
Docker image. We continue to work on porting more NN architectures to
Supervisely. 

We also have a detailed guide on how to [make your
own neural net plugin](./help/tutorials/03_custom_neural_net_plugin/custom_nn_plugin.md), so you
do not have to depend on anyone else to be able to use your favorite
architecture.

Here are the neural network architectures available out of the box today:

* [CNN-LSTM-CTC](./nn/cnn_lstm_ctc) (OCR)
* [DeepLab v3 Plus](./plugins/nn/deeplab_v3plus) (segmentation)
* [EAST](./plugins/nn/east) (text detection)
* [ICNet](./plugins/nn/icnet) (segmentation)
* [Mask RCNN](./plugins/nn/mask_rcnn_matterport) (detection and segmentation)
* [ResNet](./plugins/nn/resnet_classifier) (classification)
* [Models from TensorFlow Object Detection](./plugins/nn/tf_object_detection) (detection)
* [U-Net](./plugins/nn/unet_v2) (segmentation)
* [YOLO v3](./plugins/nn/yolo_v3) (detection)

Read [here](https://docs.new.supervise.ly/neural-networks/overview/overview/) how to run training or inference with this models.

For all source implementations of NNs the original authors are retaining all their original rights.

# <img src="https://i.imgur.com/5ikY4mA.png"> Import Plugins

We provide import plugins for over 30 popular dataset formats out of the box.
You can also leverage our [Python SDK](#python-sdk) to implement a new plugin for
your custom format.

# <img src="https://i.imgur.com/9s0CCP2.png"> Data Transformation Language

Data Transformation Language allows to automate complicated pipelines of data transformation. Different actions determined by *DTL layers* may be applied to images and annotations. In details it is described [here](https://docs.new.supervise.ly/export/).

# Resources

- [Supervise.ly](https://supervise.ly) - Website
- [Medium](https://medium.com/@deepsystems) - Our technical blog.
  Regular updates on how to use state of the art models and solve practical
  data science problems with Supervisely.
- [Tutorials and Cookbooks](./help) in this repository.

# The Team
