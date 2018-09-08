# Description
This repo contains a Jupyter Notebook that makes inference of unet model

# How to run
Execute the following commands:
``` 
git clone https://github.com/supervisely/supervisely.git
tutorials/03_slwin_inference/docker/run.sh
```
to build docker image and run the container. Then, within the container:
``` 
jupyter notebook --allow-root --ip=0.0.0.0
```
Your token will be shown in terminal.
After that, run in browser: 
```
http://localhost:8888/?token=your_token
```
