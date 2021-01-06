# Creating a Supervisely plugin

This tutorial walks you through creating a new Supervisely plugin. In Supervisely, plugins are Docker images that are launched on worker machines by the *[Supervisely agent](../../../agent/README.md)*.

Here we cover the steps of making a new Supervisely plugin from scratch:

1. Use the special code layout required by Supervisely web instance.
2. Pack the resulting code into a Docker image and publish to a public repository.
3. Tell the Supervisely web instance about your new Docker image.

## Plugin file layout

Our web system and build script require the following code layout:

```
your_plugin_root_dir
│
├── src
│   ├── main.py  (entry point)
│   └── (... other source files ...)
├── supervisely_lib (optional)
│   └── ...
├── Dockerfile
├── LICENSE
├── plugin_info.json
├── predefined_run_configs.json (optional)
├── README.md
└── VERSION
```

### Sources

All your source code must be inside the `src` subdirectory.

Note that the entrypoint must be a Python file named `main.py` (except for neural network plugins, which use `train.py` and `inference.py` entrypoints).

Typically you would want to use our [Python SDK](https://github.com/supervisely/supervisely/tree/master/supervisely_lib) for processing data in Supervisely formats, logging in a compatible format that the web instance can interpret etc. Copy the library to a `supervisely_lib` subdirectory so that it is available when building a Docker image.

### Dockerfile

Supervisely relies on Docker images to store the plugin exectutables. If you are not familiar with Docker basics, [Docker get started guide](https://docs.docker.com/get-started/) is a good starting point.

It is important to have your Python entrypoints be in the `/workdir/src` directory for our build script to work. If you keep to the recommended file layout, then in your `Dockerfile` you can simply use
```
COPY . /workdir
```
to put all your Python code inside the image. Also you need to set up `PYTHONPATH` environment variable to make all the source code visible to the Python interpeter:
```
ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
```

### License

It’s important for every plugin to include a license. This tells users who install your package the terms under which they can use your package. For help picking a license, see https://choosealicense.com/. Once you have chosen a license, open LICENSE and enter the license text.


### Plugin info

`plugin_info.json` describes the name, description and type of your plugin. It must contain a JSON object of the following format:
```python
{
	"title": "<Plugin name>",
	"description": "<Plugin description>",
	"type": "<Plugin type>"
}
```

The name and description will be displayed to the user in Supervisely web interface. The plugin type is interpreted by the system to determine which kind of inputs and outputs are expected for the plugin, and which entry points should be used.

Supervisely supports the following plugin types:
* `import` plugins are intended for converting labeled datasets from another format to Supervisely format and uploading the converted data to the Supervisely web instance. These plugins will be available in the "Import" page.
* `architecture` plugins are intended for implementation of neural network models. They should contain the code for training and inference, taking in projects in Supervisely format as input.


### Readme

The README file is a Markdown file that will be available on Plugin Page. Example:

![](https://i.imgur.com/YjNwmiP.png)


### Version

Version file contains the name of the docker image (without docker registry name) and its tag in the following format `<docker image>:<tag>`. The plugin name and docker registry identify a plugin in the Supervisely web instance. The version tag (recorded as a Docker tag by the build script) lets you switch between different versions of the plugin. Switching to a different version may be useful if a plugin code update introduced a bug and you want to roll back to one of the previous versions. Example `VERSION` file:
```
some/prefix/nn-icnet:4.0.0
```

### Predefined run configs

If your plugin takes a specific configuration as input, it is convenient to prepare some default examples in `predefined_run_configs.json` file with the following structure:

```python
[
  {
    "title": "<template title>",
    "type": "<type of the plugin or mode (train/inference) in case of neural networks>",
    "config": {"custom configuration": "is here"}
  },
  {
    "title": "<another template title>",
    "type": "<type of the plugin or mode>",
    "config": {"different custom configuration": "is here"}
  }
]
```


## Build and publish the plugin Docker image

To make your plugin available to the Supervisely web instance, you need to publish the plugin to a public Docker repository. If you do not have your own repository set up, the free [Docker Hub](https://docs.docker.com/docker-hub/) is a good default option.

Once you figure out your Docker repository setup, build the plugin docker image by executing the [`build_plugin.sh`](./build_plugin.sh) script:

```sh
./build_plugin.sh your.docker.registry path/to/your_plugin_root_dir
```

The script will print out build status information, ending with the full name of the plugin Docker image. If the build was successful, publish your image to the public repository. If the `VERSION` file contains `some/prefix/nn-icnet:4.0.0` as in our example, the command would be:
```sh
docker push your.docker.registry/some/prefix/nn-icnet:4.0.0
```

## Tell the Supervisely web instance about your new plugin

Go to "Plugins" page and press "Add" button:

![](https://i.imgur.com/uvBF7y2.png)

Enter plugin title and docker image and press "Create" button:

![](https://i.imgur.com/DJsuyJ4.png)

As a result new plugin will appear in your "Plugins" page:

![](https://i.imgur.com/YjNwmiP.png)
