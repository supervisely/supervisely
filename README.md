<h1 align="center">
  <br>
  <a href="https://supervise.ly"><img alt="Supervisely" title="Supervisely" src="https://i.imgur.com/o8QhNOv.png" width="250"></a>
  <br>
  Supervisely
  <br>
</h1>

<h4 align="center">AI for everyone! Neural networks, tools and a library we use in <a href="https://supervise.ly">Supervisely</a>.</h4>

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
  <a href="#dtl">DTL</a> •  
  <a href="#neural-networks">Neural Networks</a> •
  <a href="#library">Library</a> •
  <a href="#related">Related</a>
</p>

![screenshot](https://i.imgur.com/5dzQrrA.gif)

## Introduction

Supervisely is a web platform where you can find everything you need to build Deep Learning solutions within a single environment.

We learn a lot from our awesome comunity and want to give something back. Here you can find our python code we use to develop models and tools like DTL and also a source code for agent you deploy on your PC.

## Agent

Supervisely Agent is a simple open-sourced task manager available as a Docker image.

Agent connects to Supervisely API and so you can run tasks like import, DTL, training and inference on a connected computer — host.

<p align="center">
<img src="https://docs.new.supervise.ly/images/cluster/agent-diagramm.png" alt="Deploying agent to Supervisely" width="400" />
</p>

Internally, we use [protobuf](https://developers.google.com/protocol-buffers/) for communication with server. Check out [documentation](https://docs.new.supervise.ly/cluster/overview/) on how to deploy a new agent.

## DTL

Data Transformation Language allows to automate complicated pipelines of data transformation. Different actions determined by *DTL layers* may be applied to images and annotations. In details it is described [here](https://docs.new.supervise.ly/export/).

## Neural Networks

A number of different Neural Networks (NNs) is provided in Supervisely. NN architectures are available as separate Docker images.

* [U-Net](https://github.com/supervisely/supervisely/tree/master/nn/unet_v2)
* [DeepLab](https://github.com/supervisely/supervisely/tree/master/nn/deeplab_v3plus)
* [Mask R-CNN](https://github.com/supervisely/supervisely/tree/master/nn/tf_mask)
* [YOLO](https://github.com/supervisely/supervisely/tree/master/nn/yolo_v3)
* [SSD MobileNet](https://github.com/supervisely/supervisely/tree/master/nn/tf_ssd)
* [Faster R-CNN](https://github.com/supervisely/supervisely/tree/master/nn/faster_rcnn)
* [ICNet](https://github.com/supervisely/supervisely/tree/master/nn/icnet) (based on [this implementation](https://github.com/hellochick/ICNet-tensorflow))
* [PSPNet](https://github.com/supervisely/supervisely/tree/master/nn/pspnet) (based on [this implementation](https://github.com/hellochick/PSPNet-tensorflow))

Read [here](https://docs.new.supervise.ly/neural-networks/overview/overview/) how to run training or inference with this models.

For all source implementations of NNs authors are retaining their original rights.

## Library

Supervisely Lib contains Python code which is useful to process data in Supervisely format and to integrate new NNs with Supervisely.

Key features:
 * Read, modify and write Supervisely projects on your disk;
 * Work with figures (annotations);
 * Modify existing implementations of NNs or to create new ones which are compatible with Supervisely;

Reference may be found [here](https://docs.new.supervise.ly/sly-lib/).

## Related

- [Supervise.ly](https://supervise.ly) - Website
- [Medium](https://medium.com/@deepsystems) - Recent tutorials on how to use SotA models
- [Tutorials](https://github.com/DeepSystems/supervisely-tutorials) - Repo with tutorials sources and link to a related blog posts

