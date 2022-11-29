<h1 align="center">
  <a href="https://supervise.ly"><img alt="Supervisely" title="Supervisely" src="https://i.imgur.com/B276eMS.png"></a>
</h1>

<h3 align="center">
<a href="https://supervise.ly">Computer Vision Platform</a>, 
<a href="https://ecosystem.supervise.ly/">Open Ecosystem of Apps</a>,
<a href="https://developer.supervise.ly/">SDK for Python</a>
</h3>

<p align="center">
  <a href="https://pypi.org/project/supervisely" target="_blank">
    <img src="https://static.pepy.tech/personalized-badge/supervisely?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20installs" alt="Package version">
  </a>
  <a href="https://hub.docker.com/r/supervisely/agent/tags" target="_blank">
    <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/supervisely/agent?label=docker%20pulls%20-%20supervisely%2Fagent">
  </a>
  <br/>
  <a href="https://pypi.org/project/supervisely" target="_blank">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/supervisely?color=4ec528">
  </a>
  <a href="https://supervise.ly/slack" target="_blank"> <img src="https://img.shields.io/badge/slack-chat-green.svg?logo=slack&color=4ec528" alt="Slack">
  </a>
  <a href="https://pypi.org/project/supervisely" target="_blank"> 
    <img src="https://img.shields.io/pypi/v/supervisely?color=4ec528&label=pypi%20package" alt="Package version"> 
  </a>
  <a href="https://developer.supervise.ly" target="_blank">
    <img src="https://readthedocs.org/projects/supervisely/badge/?version=stable&color=4ec528">
  </a>
</p>

---

**Website**: [https://supervise.ly](https://supervise.ly/)

**Supervisely Ecosystem**: [https://ecosystem.supervise.ly](https://ecosystem.supervise.ly/)

**Dev Documentation**: [https://developer.supervise.ly](https://developer.supervise.ly/)

**Source Code of SDK for Python**: [https://github.com/supervisely/supervisely](https://github.com/supervisely/supervisely)

**Supervisely Ecosystem on GitHub**: [https://github.com/supervisely-ecosystem](https://github.com/supervisely-ecosystem)

---

## Table of contents

- [Introduction](#introduction)
  - [Supervisely Platform 🔥](#supervisely-platform-)
  - [Supervisely Ecosystem 🎉](#supervisely-ecosystem-)
- [Development 🧑‍💻](#development-)
  - [What developers can do](#what-developers-can-do)
    - [Level 1. HTTP REST API](#level-1-http-rest-api)
    - [Level 2. Python scripts for automation and integration](#level-2-python-scripts-for-automation-and-integration)
    - [Level 3. Headless apps (without UI)](#level-3-headless-apps-without-ui)
    - [Level 4. Apps with interactive UIs](#level-4-apps-with-interactive-uis)
    - [Level 5. Apps with UI integrated into labeling tools](#level-5-apps-with-ui-integrated-into-labeling-tools)
  - [Principles 🧭](#principles-)
- [Main features 💎](#main-features-)
  - [Start in a minute](#start-in-a-minute)
  - [Magically simple API](#magically-simple-api)
  - [Customization is everywhere](#customization-is-everywhere)
  - [Interactive GUI is a game-changer](#interactive-gui-is-a-game-changer)
  - [Develop fast with ready UI widgets](#develop-fast-with-ready-ui-widgets)
  - [Convenient debugging](#convenient-debugging)
  - [Apps can be both private and public](#apps-can-be-both-private-and-public)
  - [Single-click deployment](#single-click-deployment)
  - [Reliable versioning - releases and branches](#reliable-versioning---releases-and-branches)
  - [Supports both Github and Gitlab](#supports-both-github-and-gitlab)
  - [App is just a web server, use any technology you love](#app-is-just-a-web-server-use-any-technology-you-love)
  - [Built-in cloud development environment (coming soon)](#built-in-cloud-development-environment-coming-soon)
  - [Trusted by Fortune 500. Used by 65 000 researchers, developers, and companies worldwide](#trusted-by-fortune-500-used-by-65-000-researchers-developers-and-companies-worldwide)
- [Community 🌎](#community-)
    - [Have an idea or ask for help?](#have-an-idea-or-ask-for-help)
- [Contribution 👏](#contribution-)
- [Partnership 🤝](#partnership-)


## Introduction

Every company wants to be sure that its current and future AI tasks are solvable.

The main issue with most solutions on the market is that they build as products. It's a black box developing by some company you don't really have an impact on. As soon as your requirements go beyond basic features offered and you want to customize your experience, add something that is not in line with the software owner development plans or won't benefit other customers, you're out of luck.

That is why **Supervisely is building a platform** instead of a product.

### [Supervisely Platform 🔥](https://supervise.ly/)

<a href="https://supervise.ly/">
  <img src="https://user-images.githubusercontent.com/73014155/178843741-996aff24-7ceb-4e3e-88fe-1c19ccd9a757.png" style="max-width:100%;"
  alt="Supervisely Platform">
</a>

You can think of [Supervisely](https://supervise.ly/) as an Operating System available via Web Browser to help you solve Computer Vision tasks. The idea is to unify all the relevant tools within a single [Ecosystem](https://ecosystem.supervise.ly/) of apps, tools, UI widgets and services that may be needed to make the AI development process as smooth and fast as possible.

More concretely, Supervisely includes the following functionality:

* Data labeling for images, videos, 3D point cloud and volumetric medical images (dicom)
* Data visualization and quality control
* State-Of-The-Art Deep Learning models for segmentation, detection, classification and other tasks
* Interactive tools for model performance analysis
* Specialized Deep Learning models to speed up data labeling (aka AI-assisted labeling)
* Synthetic data generation tools
* Instruments to make it easier to collaborate for data scientists, data labelers, domain experts and software engineers

### [Supervisely Ecosystem](https://ecosystem.supervise.ly/) 🎉


<a href="https://ecosystem.supervise.ly/">
  <img src="https://user-images.githubusercontent.com/73014155/178843764-a92b7ad4-0cce-40ce-b849-17b49c1e1ad3.png" style="max-width:100%;"
  alt="Supervisely Platform">
</a>

The simplicity of creating Supervisely Apps has already led to the development of [hundreds of applications](https://ecosystem.supervise.ly/), ready to be run within a single click in a web browser and get the job done.

Label your data, perform quality assurance, inspect every aspect of your data, collaborate easily, train and apply state-of-the-art neural networks, integrate custom models, automate routine tasks and more — like in a real AppStore, there should be an app for everything.

## [Development](https://developer.supervise.ly/) 🧑‍💻

Supervisely provides the foundation for integration, customization, development and running computer vision applications to address your custom tasks - just like in OS, like Windows or MacOS.

### What developers can do

There are different levels of integration, customization, and automation:

1. [HTTP REST API](#level-1-http-rest-api)
2. [Python scripts for automation and integration](#level-2-python-scripts-for-automation-and-integration)
3. [Headless apps (without UI)](#level-3-headless-apps-without-ui)
4. [Apps with interactive UIs](#level-4-apps-with-interactive-uis)
5. [Apps with UIs integrated into labeling tools](#level-5-apps-with-ui-integrated-into-labeling-tools)

#### Level 1. HTTP REST API

Supervisely has a rich [HTTP REST API](https://api.docs.supervise.ly/) that covers basically every action, you can do manually. You can use **any programming language** and **any development environment** to extend and customize your Supervisely experience.

ℹ️ For Python developers, we recommend using our [Python SDK](https://supervisely.readthedocs.io/en/latest/sdk\_packages.html) because it wraps up all API methods and can save you a lot of time with built-in error handling, network re-connection, response validation, request pagination, and so on.

<details>

<summary>cURL example</summary>

There's no easier way to kick the tires than through [cURL](http://curl.haxx.se/). If you are using an alternative client, note that you are required to send a valid header in your request.

Example:

```bash
curl -H "x-api-key: <your-token-here>" https://app.supervise.ly/public/api/v3/projects.list
```

As you can see, URL starts with `https://app.supervise.ly`. It is for Community Edition. For Enterprise Edition you have to use your custom server address.

</details>

#### Level 2. Python scripts for automation and integration

[Supervisely SDK for Python](https://supervisely.readthedocs.io/en/latest/sdk\_packages.html) is specially designed to speed up development, reduce boilerplate, and lets you do anything in a few lines of Python code with Supervisely Annotatation JSON format, communicate with the platform, import and export data, manage members, upload predictions from your models, etc.

<details>

<summary>Python SDK example</summary>

Look how it is simple to communicate with the platform from your python script.

```python
import supervisely as sly

# authenticate with your personal API token
api = sly.Api.from_env()

# create project and dataset
project = api.project.create(workspace_id=123, name="demo project")
dataset = api.dataset.create(project.id, "dataset-01")

# upload data
image_info = api.image.upload_path(dataset.id, "img.png", "/Users/max/img.png")
api.annotation.upload_path(image_info.id, "/Users/max/ann.json")

# download data
img = api.image.download_np(image_info.id)
ann = api.annotation.download_json(image_info.id)
```

</details>

#### Level 3. Headless apps (without UI)

Create python apps to automate routine and repetitive tasks, share them within your organization,  and provide an easy way to use them for end-users without coding background.  Headless apps are just python scripts that can be run from a context menu.

![run app from context menu](https://user-images.githubusercontent.com/73014155/178843779-2af6fff3-ce28-4278-a57f-f6577615b849.png)

It is simple and suitable for the most basic tasks and use-cases, for example:

* import and export in custom format ([example1](https://ecosystem.supervise.ly/apps/import-images-groups), [example2](https://ecosystem.supervise.ly/apps/export-as-masks), [example3](https://ecosystem.supervise.ly/apps/export-to-pascal-voc), [example4](https://ecosystem.supervise.ly/apps/render-video-labels-to-mp4))
* assets transformation ([example1](https://ecosystem.supervise.ly/apps/rasterize-objects-on-images), [example2](https://ecosystem.supervise.ly/apps/resize-images), [example3](https://ecosystem.supervise.ly/apps/change-video-framerate), [example4](https://ecosystem.supervise.ly/apps/convert\_ptc\_to\_ptc\_episodes))
* users management ([example1](https://ecosystem.supervise.ly/apps/invite-users-to-team-from-csv), [example2](https://ecosystem.supervise.ly/apps/create-users-from-csv), [example3](https://ecosystem.supervise.ly/apps/export-activity-as-csv))
* deploy special models for AI-assisted labeling ([example1](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Fritm-interactive-segmentation%2Fsupervisely), [example2](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Ftrans-t%2Fsupervisely%2Fserve), [example3](https://ecosystem.supervise.ly/apps/volume-interpolation))

#### Level 4. Apps with interactive UIs

Interactive interfaces and visualizations are the keys to building and improving AI solutions: from custom data labeling to model training. Such apps open up opportunities to customize Supervisely platform to any type of task in Computer Vision, implement data and models workflows that fit your organization's needs, and even build vertical solutions for specific industries on top of it.

<a href="https://ecosystem.supervise.ly/apps/dev-smart-tool-batched">
  <img src="https://user-images.githubusercontent.com/73014155/178845451-8350a6d7-f318-4f5b-a9ee-4b871016e2e4.gif" style="max-width:100%;"
  alt="[This interface is completely based on python in combination with easy-to-use Supervisely UI widgets (Batched SmartTool app for AI assisted object segmentations)">
</a>

Here are several examples:

* custom labeling interfaces with AI assistance for [images](https://ecosystem.supervise.ly/apps/dev-smart-tool-batched) and [videos](https://ecosystem.supervise.ly/apps/batched-smart-tool-for-videos)
* [interactive model performance analysis](https://ecosystem.supervise.ly/apps/semantic-segmentation-metrics-dashboard)
* [interactive NN training dashboard](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Fmmsegmentation%2Ftrain)
* [data exploration](https://ecosystem.supervise.ly/apps/action-recognition-stats) and [visualization](https://ecosystem.supervise.ly/apps/objects-thumbnails-preview-by-class) apps
* [vertical solution](https://ecosystem.supervise.ly/collections/supervisely-ecosystem%2Fgl-metric-learning%2Fsupervisely%2Fretail-collection) for labeling products on shelves in retail
* inference interfaces [in labeling tools](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Fnn-image-labeling%2Fannotation-tool); for [images](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Fnn-image-labeling%2Fproject-dataset), [videos](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) and [point clouds](https://ecosystem.supervise.ly/apps/apply-det3d-to-project-dataset); for [model ensembles](https://ecosystem.supervise.ly/apps/apply-det-and-cls-models-to-project)

#### Level 5. Apps with UI integrated into labeling tools

There is no single labeling tool that fits all tasks. Labeling tool has to be designed and customized for a specific task to make the job done in an efficient manner. Supervisely apps can be smoothly integrated into labeling tools to deliver amazing user experience (including multi tenancy) and annotation performance.

<a href="https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fgl-metric-learning%252Fsupervisely%252Flabeling-tool">
  <img src="https://user-images.githubusercontent.com/12828725/179206991-1c76f61d-b88a-4a2b-9116-d87fb1ed9d0e.png" style="max-width:100%;"
  alt="[AI assisted retail labeling app is integrated into labeling tool and can communicate with it via web sockets)">
</a>

Here are several examples:

* apps designed for custom labeling workflows ([example1](https://ecosystem.supervise.ly/apps/visual-tagging), [example2](https://ecosystem.supervise.ly/apps/review-labels-side-by-side))
* NN inference is integrated for labeling automation and model predictions analysis ([example](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Fnn-image-labeling%2Fannotation-tool))
* industry-specific labeling tool: annotation of thousands of product types on shelves with AI assistance ([retail collection](https://ecosystem.supervise.ly/collections/supervisely-ecosystem%2Fgl-metric-learning%2Fsupervisely%2Fretail-collection), [labeling app](https://ecosystem.supervise.ly/apps/ai-assisted-classification))

### Principles 🧭

Development for Supervisely builds upon these five principles:

* All in **pure Python** and build on top of your favourites libraries (opencv, requests, fastapi, pytorch, imgaug, etc ...) - easy for python developers and data scientists to build and share apps with teammates and the ML community.
* No front‑end experience is required -  build **powerful** and **interactive** web-based GUI apps using the comprehensive library of ready-to-use UI widgets and components.
* **Easy to learn, fast to code,** and **ready for production**.  SDK provides a simple and intuitive API by having complexity "under the hood". Every action can be done just in a few lines of code. You focus on your task, Supervisely will handle everything else - interfaces, databases, permissions, security, cloud or self-hosted deployment, networking, data storage, and many more. Supervisely has solid testing, documentation, and support.
* Everything is **customizable** - from labeling interfaces to neural networks. The platform has to be customized and extended to perfectly fit your tasks and requirements, not vice versa. Hundreds of examples cover every scenario and can be found in our [ecosystem of apps](https://ecosystem.supervise.ly/).
* Apps can be both **open-sourced or private**. All apps made by Supervisely team are [open-sourced](https://github.com/supervisely-ecosystem). Use them as examples, just fork and modify the way you want. At the same time, customers and community users can still develop private apps to protect their intellectual property.

## Main features 💎

- [Start in a minute](#start-in-a-minute)
- [Magically simple API](#magically-simple-api)
- [Customization is everywhere](#customization-is-everywhere)
- [Interactive GUI is a game-changer](#interactive-gui-is-a-game-changer)
- [Develop fast with ready UI widgets](#develop-fast-with-ready-ui-widgets)
- [Convenient debugging](#convenient-debugging)
- [Apps can be both private and public](#apps-can-be-both-private-and-public)
- [Single-click deployment](#single-click-deployment)
- [Reliable versioning - releases and branches](#reliable-versioning---releases-and-branches)
- [Supports both Github and Gitlab](#supports-both-github-and-gitlab)
- [App is just a web server, use any technology you love](#app-is-just-a-web-server-use-any-technology-you-love)
- [Built-in cloud development environment (coming soon)](#built-in-cloud-development-environment-coming-soon)
- [Trusted by Fortune 500. Used by 65 000 researchers, developers, and companies worldwide](#trusted-by-fortune-500-used-by-65-000-researchers-developers-and-companies-worldwide)

### Start in a minute

Supervisely's open-source SDK and app framework are straightforward to get started with. It’s just a matter of:

```
pip install supervisely
```

### Magically simple API

[Supervisely SDK for Python](https://supervisely.readthedocs.io/en/latest/sdk\_packages.html) is simple, intuitive, and can save you hours. Reduce boilerplate and build custom integrations in a few lines of code. It has never been so easy to communicate with the platform from python.

```python
# authenticate with your personal API token
api = sly.Api.from_env()

# create project and dataset
project = api.project.create(workspace_id=123, name="demo project")
dataset = api.dataset.create(project.id, "dataset-01")

# upload data
image_info = api.image.upload_path(dataset.id, "img.png", "/Users/max/img.png")
api.annotation.upload_path(image_info.id, "/Users/max/ann.json")

# download data
img = api.image.download_np(image_info.id)
ann = api.annotation.download_json(image_info.id)
```

### Customization is everywhere

Customization is the only way to cover all tasks in Computer Vision. Supervisely allows to customizing everything from labeling interfaces and context menus to training dashboards and inference interfaces. Check out our [Ecosystem of apps](https://ecosystem.supervise.ly/) to find inspiration and examples for your next ML tool.

### Interactive GUI is a game-changer

The majority of Python programs are "command line" based. While highly experienced programmers don't have problems with it, other tech people and end-users do.  This creates a digital divide, a "GUI Gap".  App with graphic user interface (GUI) becomes more approachable and easy to use to a wider audience. And finally, some tasks are impossible to solve without a GUI at all.

Imagine, how it will be great if all ML tools and repositories have an interactive GUI with the RUN button ▶️. It will take minutes to start working with a top Deep Learning framework instead of spending weeks running it on your data.  

🎯 Our ambitious goal is to make it possible.


<a href="https://ecosystem.supervise.ly/apps/semantic-segmentation-metrics-dashboard">
  <img src="https://user-images.githubusercontent.com/73014155/178846370-ae86dd3c-e08d-4df2-871b-d342bf7ba370.gif" style="max-width:100%;"
  alt="Semantic segmentation metrics app">
</a>


### Develop fast with ready UI widgets

Hundreds of interactive UI widgets and components are ready for you. Just add to your program and populate with the data. Python devs don't need to have any front‑end experience, in our developer portal you will find needed guides, examples, and tutorials. We support the following UI widgets:

1. [Widgets made by Supervisely](https://ecosystem.supervise.ly/docs/grid-gallery) specifically for computer vision tasks, like rendering galleries of images with annotations, playing videos forward and backward with labels, interactive confusion matrices, tables, charts, ...
2. [Element widgets](https://element.eleme.io/1.4/#/en-US/component/button) - Vue 2.0 based component library
3. [Plotly](https://plotly.com/python/) Graphing Library for Python
4. You can develop your own UI widgets ([example](https://github.com/supervisely-ecosystem/dev-smart-tool-batched/blob/master/static/smarttool.js))

Supervisely team makes most of its apps publically available on [GitHub](https://github.com/supervisely-ecosystem). Use them as examples for your future apps: fork, modify, and copy-paste code snippets.

### Convenient debugging

Supervisely is made by data scientists for data scientists. We trying to lower barriers and make a friendly development environment. Especially we care about debugging as one of the most crucial steps.

Even in complex scenarios, like developing a GUI app integrated into a labeling tool, we keep it simple - use breakpoints in your favorite IDE to catch callbacks, step through the program and see live updates without page reload. As simple as that! Supervisely handles everything else -  WebSockets, authentication, Redis, RabitMQ, Postgres, ...

Watch the video below, how we debug [the app](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Fnn-image-labeling%2Fannotation-tool) that applies NN right inside the labeling interface.

<a href="https://youtu.be/fOnyL8YHOBM">
    <img src="https://user-images.githubusercontent.com/12828725/179207006-bcdd0922-21c1-4958-86e7-d532fbf7c974.png" style="max-width:100%;">
</a>

### Apps can be both private and public

All apps made by Supervisely team are [open-source](https://github.com/supervisely-ecosystem). Use them as examples: find on [GitHub](https://github.com/supervisely-ecosystem), fork and modify them the way you want. At the same time, customers and community users can still develop private apps to protect their intellectual property.

<a href="https://youtu.be/Kyuc-lZu_tg">
    <img src="https://user-images.githubusercontent.com/12828725/179207014-55659b39-0f58-42db-96e3-8063f1e6ad5d.png" style="max-width:100%;">
</a>

### Single-click deployment

Supervisely app is a git repository. Just provide the link to your git repo, Supervisely will handle everything else. Now you can press `Run` button in front of your app and start it on any computer with [Supervisely Agent](https://youtu.be/aDqQiYycqyk).

### Reliable versioning - releases and branches

Users run your app on the latest stable release, and you can develop and test new features in parallel - just use git releases and branches. Supervisely automatically pull updates from git, even if the new version of an app has a bug, don't worry - users can select and run the previous version in a click.


<a href="https://youtu.be/ngoHfM98R8k">
    <img src="https://user-images.githubusercontent.com/12828725/179207015-d5f839a6-907b-4469-9f86-950ee348024e.png" style="max-width:100%;">
</a>

### Supports both Github and Gitlab

Since Supervisely app is just a git repository, we support public and private repos from the most popular hosting platforms in the world - GitHub and GitLab.

### App is just a web server, use any technology you love 

Supervisely SDK for Python provides the simplest way for python developers and data scientists to build interactive GUI apps of any complexity. Python is a recommended language for developing Supervisely apps, but not the only one. You can use any language or any technology you love, any web server can be deployed on top of the platform.

For example, even [Visual Studio Code for web](https://github.com/coder/code-server) can be run as an app (see video below). 

### Built-in cloud development environment (coming soon)

In addition to the common way of development in your favorite IDE on your local computer or laptop, cloud development support will be integrated into Supervisely and **released soon** to speed up development, standardize dev environments, and lower barriers for beginners. 

How will it work? Just connect your computer to your Supervisely instance and run IDE app ([JupyterLab](https://jupyter.org/) and [Visual Studio Code for web](https://github.com/coder/code-server)) to start coding in a minute. We will provide a large number of template apps that cover the most popular use cases.


<a href="https://youtu.be/ptHJsdolHHk">
    <img src="https://user-images.githubusercontent.com/73014155/178956713-0de05a39-3ecc-41b2-a46e-54d3a23d4e64.png" style="max-width:100%;">
</a>

### Trusted by Fortune 500. Used by 65 000 researchers, developers, and companies worldwide

![](https://user-images.githubusercontent.com/106374579/204510683-4aaa1e11-e934-4268-8365-f140028508d0.png)

Supervisely helps companies and researchers all over the world to build their computer vision solutions in various industries from self-driving and agriculture to medicine. Join our [Community Edition](https://app.supervise.ly/) or request [Enterprise Edition](https://supervise.ly/enterprise) for your organization.

## Community 🌎

Join our constantly growing [Supervisely community](https://app.supervise.ly/) with more than 65k+ users.

#### Have an idea or ask for help?

If you have any questions, ideas or feedback please:

1. [Suggest a feature or idea](https://ideas.supervise.ly/), or [give a technical feedback ](https://github.com/supervisely/supervisely/issues)
2. [Join our slack](https://supervise.ly/slack)
3. [Contact us](https://supervise.ly/contact-us)

Your feedback 👍 helps us a lot and we appreciate it

## Contribution 👏

Want to help us bring Computer Vision R\&D to the next level? We encourage you to participate and speed up R\&D for thousands of researchers by

* building and expanding Supervisely Ecosystem with us
* integrating to Supervisley and sharing your ML tools and research with the entire ML community

## Partnership 🤝

We are happy to expand and increase the value of Supervisely Ecosystem with additional technological partners, researchers, developers, and value-added resellers.

Feel free to [contact us](https://supervise.ly/contact-us) if you have

* ML service or product
* unique domain expertise
* vertical solution
* valuable repositories and tools that solve the task
* custom NN models and data

Let's discuss the ways of working together, particularly if we have joint interests, technologies and  customers.
