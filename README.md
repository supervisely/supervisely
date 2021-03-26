<h1 align="center">
  <br>
  <a href="https://supervise.ly"><img alt="Supervisely" title="Supervisely" src="https://i.imgur.com/bFuEQ4K.png" width="250"></a>
  <br>
  Supervisely
  <br>
</h1>

<h4 align="center">AI for everyone! Neural networks, tools and Python SDK we use with <a href="https://supervise.ly">Supervisely</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/uptimerobot/status/m778791913-8b2f81d0f1c83da85158e2a5.svg">
  <img src="https://img.shields.io/uptimerobot/ratio/m778791913-8b2f81d0f1c83da85158e2a5.svg">
  <img src="https://img.shields.io/github/repo-size/supervisely/supervisely.svg">
  <img src="https://img.shields.io/github/languages/top/supervisely/supervisely.svg">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
</p>


<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#agent">Agent</a> •
  <a href="#neural-networks">Neural Networks</a> •
  <a href="#import-plugins">Import Plugins</a> •
  <a href="#python-sdk">Python SDK</a> •
  <a href="#data-transformation-language">Data Transformations</a> •
  <a href="#resources">Resources</a>
</p>

![screenshot](https://i.imgur.com/5dzQrrA.gif)

## Introduction

Supervisely is a web platform where you can find everything you need to build Deep Learning solutions within a single environment.

We learn a lot from our awesome community and want to give something back.
Here you can find the Python SDK we use to implement neural network models,
import plugins for custom data formats, and tools like the Data Transformation
Language. You can also find the source code for the agent to turn your PC into a
worker node to deploy your neural nets. 

## Agent

Supervisely Agent is a simple open-source task manager available as a Docker image.

The Agent runs on a worker node. It connects to the Supervisely WEB instance and
listens for the tasks (like neural net training request) to run. It handles
downloading and uploading data to the web instance, sets up proper environments
for the specific tasks to run, and keeps track of progress, successes and
failures of individual tasks. By deploying the agent on worker machine you bring
up a virtual computing claster that your team can run their tasks on from the
Supervisely web instance.

<p align="center">
<img src="https://gblobscdn.gitbook.com/assets%2F-M4BHwRbuyIoH-xoF3Gv%2F-M5JQKcQ0OcHshO-q9Kz%2F-M5JQLtrAGKs7RWLDVdA%2Fagent-diagramm.png" alt="Deploying agent to Supervisely" width="400" />
</p>

Check out [explanation](https://github.com/supervisely/supervisely/tree/master/agent) on how agent works and [documentation](https://docs.supervise.ly/customization/agents) on how to deploy a new agent.

## Neural Networks

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

## Import plugins

We provide import plugins for over 30 popular dataset formats out of the box.
You can also leverage our [Python SDK](#python-sdk) to implement a new plugin for
your custom format.

## Python SDK

We have organized most of the common functionality for processing data in
[Supervisely format](https://docs.supervise.ly/ann_format/) and for training and
inference with neural networks into the [Python SDK](./supervisely_lib). Our stock plugins rely on the SDK
extensively, and we believe the SDK will be also valuable to the community.

The  SDK not only wraps all the low-level details of handling the data and
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

The only prerequisites are `Python` >= 3.6 and `pip`.

Tip: `opencv-python` may require `apt-get install libgtk2.0-dev` Or use pre-built Docker image which can be found on Docker Hub:

```docker pull supervisely/base-py```

The corresponding `Dockerfile` can be found in `base_images` directory. 



## Data Transformation Language

Data Transformation Language allows to automate complicated pipelines of data transformation. Different actions determined by *DTL layers* may be applied to images and annotations. In details it is described [here](https://docs.new.supervise.ly/export/).

## Resources

- [Supervise.ly](https://supervise.ly) - Website
- [Medium](https://medium.com/@deepsystems) - Our technical blog.
  Regular updates on how to use state of the art models and solve practical
  data science problems with Supervisely.
- [Tutorials and Cookbooks](./help) in this repository.
