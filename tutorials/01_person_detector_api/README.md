# Description
This repo contains a Jupyter Notebook that makes API requests to person detection model. Step by step guide on how to run a person detector is given in this blog post

# How to run
Clone this repo, then
``` 
cd docker
./run.sh 
```
to pull docker image and run the container. Then, within the container:
``` 
jupyter notebook --allow-root --ip=0.0.0.0
```
Your token will be shown in terminal.
After that, run in browser: 
```
http://localhost:8888/?token=your_token
```
# Example
After running deploy_faster-rcnn_as_api.ipynb, you get the following results:
![Image description](link-to-image)
