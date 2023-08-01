# Docker Images for Supervisely Applications

This folder contains Dockerfiles for Supervisely applications.
Each Dockerfile is based on the `supervisely/base-py-sdk` image and contains all the required dependencies for the corresponding Supervisely applications category in [Supervisely Ecosystem](https://ecosystem.supervisely.com/).

Docker images are automatically updated through the [GitHub action](.github/workflows/sdk_pip_docker.yml) when a new Supervisely SDK version is released. Docker image tags are the same as the Supervisely SDK version.

## Available Dockerfiles

* [Collaboration](collaboration/Dockerfile)
* [Data Operations](data-operations/Dockerfile)
* [Development](development/Dockerfile)
* [Import-Export](import-export/Dockerfile)
* [Labeling](labeling/Dockerfile)
* [Synthetic](synthetic/Dockerfile)
* [System](system/Dockerfile)
* [Visualization-Stats](visualization-stats/Dockerfile)
