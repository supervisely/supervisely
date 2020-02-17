# Model deployment

After you have trained a neural network (or selected already pre-trained one from Explore page) you can apply it within Supervisely on a Project (via "Test" button) or deploy it as API.

We support two methods of API deployment: through Web UI or completely stand alone.


## Method 1: Through UI

Easiest method, fully managed by Supervisely.

1. Go to the "Neural Networks" page. Click "three dots" icon on the model you want to deploy and select "Deploy". Deployment settings dialog will be opened
![](images/deploy_model_context.png)

2. Here you can setup deployment settings. After clicking "Submit" button and you will be redirected to the Cluster > Tasks page
![](images/deploy_model_settings.png)

3. Wait until value in the "output" column will be changed to "Deployed", click on "three dots" and select "Deploy API Info"
![](images/deploy_model_task_context.png)

4. Here you can see deployed model usage example though CURL or Python. You also can just drag'n'drop image to test your deployed model right in the browser
![](images/deploy_model_task_dialog.png)

## Method 2: Stand alone

Choose this method if you want to deploy model in production environment without Supervisely. 

**Important notice:** all steps described below are applicable to all Neural Networks that are integrated to Supervisely platform. In our examples we use YOLO v3 COCO. 

### Run HTTP server for inference as Docker container

1. Obtain docker image of the model. You can find it on right side at the model page
![](images/model_docker_image.png)

Command template:

```sh
docker pull <docker image name>
```

For example:
```sh
docker pull supervisely/nn-yolo-v3
```

2. Download model weights and extract .tar archive
![](images/deploy_model_download.png)

3. Then run following in your terminal

```sh
docker run --rm -it \
    --runtime=nvidia \
    -p <port of the host machine>:5000 \
    -v '<folder with extracted model weights>:/sly_task_data/model' \
    -env GPU_DEVICE=<device_id> \
    <model docker image> \
    python /workdir/src/rest_inference.py
```

If your machine has several GPUs, environment variable ```GPU_DEVICE``` is used to explicitly define the device id for model placement. It is the optional field (default value is 0). Especially, this parameter is helpful when you deploy TensorFlow-based models. By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process (link to [docs](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)). 


For example:

```sh
docker run --rm -it \
    --runtime=nvidia \
    -p 5000:5000 \
    -v '/home/ds/Downloads/YOLO v3 (COCO):/sly_task_data/model' \
    --env GPU_DEVICE=0 \
    supervisely/nn-yolo-v3 \
    python /workdir/src/rest_inference.py
``` 

Server is started on port 5000 inside a docker container. If you want to change the port, just bind container port 5000 to some other port (e.g. 7777) on your host machine. In such case the command is:

```sh
docker run --rm -it \
    --runtime=nvidia \
    -p 7777:5000 \
    -v '/home/ds/Downloads/YOLO v3 (COCO):/sly_task_data/model' \
    --env GPU_DEVICE=0 \
    supervisely/nn-yolo-v3 \
    python /workdir/src/rest_inference.py
```


## How to send HTTP requests

There are few ways how you can send requests:

1. Using Supervisely Python SDK. A bunch of examples can be founded in Explore->Notebooks. For example: [
Guide #04: neural network inference](https://supervise.ly/explore/notebooks/guide-04-neural-network-inference-20/overview)

2. Using CURL and bash (see below)

3. Implement on your favorite language. The model is deployed as HTTP web server, so you can implement client on any language. Python example (without Supervisely-SDK) is provided at the end of this tutorial.


### Get classes and tags that model produces 

If you want to obtain model classes and tags, there is a special method ```/model/get_output_meta```. Here is the template:


```sh
curl -H "Content-Type: multipart/form-data" -X POST \
     -F 'meta=<project meta sting in json format (optional field)>' \
     -F 'mode=<inference mode string in json format (optional field)>' \
     <ip-address of host machine>:<port of host machine, default 5000>/model/get_output_meta
```

The intuition behind optional fields ```meta``` and ```mode``` is [here](https://github.com/supervisely/supervisely/blob/master/help/jupyterlab_scripts/src/tutorials/09_detection_segmentation_pipeline/detection_segmentation_pipeline.ipynb). Examples are presented in next section.   

### Inference
All neural networks in Supervisely support several inference modes: full image, sliding window, region of interest (ROI) and bounding boxes mode. Lear more [here](https://docs.supervise.ly/neural-networks/configs/inference_config/), [here](https://github.com/supervisely/supervisely/blob/master/help/jupyterlab_scripts/src/tutorials/06_inference_modes/inference_modes.ipynb) and [here](https://github.com/supervisely/supervisely/blob/master/help/jupyterlab_scripts/src/tutorials/09_detection_segmentation_pipeline/detection_segmentation_pipeline.ipynb)


### Full image inference

Lets feed the entire image to deployed neural network. Image will be automatically resized to the input resolution of NN. The result will be returned in [Supervisely JSON format](https://docs.supervise.ly/ann_format/).


CURL template:
```
curl -X POST -F "image=@</path/to/image.png>" <ip address if the host machine>:<port of the host machine>/model/inference
```

For example, let's apply already deployed model (YOLO v3 COCO) to the image ```ties.jpg```

```sh
curl -X POST -F "image=@./images/ties.jpg" 0.0.0.0:5000/model/inference
```


The result is the following:


| Input image              |  Visualized prediction   |
:-------------------------:|:-------------------------:
<img src="./images/ties.jpg" width="600"> | <img src="https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/assets/projects/images/K/x/Tg/fOxYINjPhfmzxUMgtkiLMQvpEIxR8TNNzfHm6h2kJtZ5pSUew8TsC0tvO6JIZqUdq9mOVw0mMctDFTFlrFRsidAF98JfcZ97cjvXHzLjGNwTo60kZ8J7zHeB7FUs.png" width="600">



### Sliding window inference
CURL template:

```sh
curl -H "Content-Type: multipart/form-data" -X POST \
     -F 'mode=<inference mode string in json format (optional field)>' \
     -X POST -F "image=@</path/to/image.png>" \
     <ip address if the host machine>:<port of the host machine>/model/inference 
```

For example we have a big image. The raw json file ```sliding_window_mode_example.json``` is the following:
```json
{
  "name": "sliding_window_det",
  "window": {
    "width": 1000,
    "height": 1000
  },
  "min_overlap": {
    "x": 200,
    "y": 200
  },
  "save": false,
  "class_name": "sliding_window_bbox",
  "nms_after": {
    "enable": true,
    "iou_threshold": 0.2,
    "confidence_tag_name": "confidence"
  },
  "model_classes": {
    "add_suffix": "_det",
    "save_classes": [
      "tie"
    ]
  },
  "model_tags": {
    "add_suffix": "_det",
    "save_names": "__all__"
  }
}
```

Here is the explanation:

```py
{
    "name": "sliding_window_det",

    # Sliding window parameters.

    # Width and height in pixels.
    # Cannot be larger than the original image.
    "window": {
      "width": 1000,
      "height": 1000,
    },

    # Minimum overlap for each dimension. The last
    # window in every dimension may have higher overlap
    # with the previous one if necessary to fit the whole
    # window within the original image.
    "min_overlap": {
      "x": 200,
      "y": 200,
    },

    # Whether to save each sliding window instance as a
    # bounding box rectangle.
    "save": False,

    # If saving the sliding window bounding boxes, which
    # class name to use.
    "class_name": 'sliding_window_bbox',

    "nms_after": {

      # Whether to run non-maximum suppression after accumulating
      # all the detection results from the sliding windows.
      "enable": True,

      # Intersection over union threshold above which the same-class
      # detection labels are considered to be significantly inersected
      # for non-maximum suppression.
      "iou_threshold": 0.2,

      # Tag name from which to read detection confidence by which we
      # rank the detections. This tag must be added by the model to
      # every detection label.
      "confidence_tag_name": "confidence"
    },
    
    # Class renaming and filtering settings.
    # See "Full image inference" example for details.
    "model_classes": {
      "add_suffix": "_model",
      "save_classes": ["tie"]
    },
    
    "model_tags": {
      "add_suffix": "_model",
      "save_names": "__all__"
    }
}
```

Let's apply model:

```sh
curl -H "Content-Type: multipart/form-data" -X POST -F "image=@./images/big_image.jpg" -F mode="$(cat sliding_window_mode_example.json)" 0.0.0.0:5000/model/inference
```


| Input image              |  Visualized prediction   |
:-------------------------:|:-------------------------:
<img src="https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/assets/projects/images/J/3/qA/awJVhndwAUgoLdo2eAxO4N7QewaLZjCug44npXfvV4YeJCPkvegRHncOawDcLRoPPPsVox0eDE2imyfsD7s8XORExbPURdHrPnVm4PNIQD321l1ddUlwRhcelSUV.png" width="600"> | <img src="https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/assets/projects/images/S/p/RV/rb1AAS7dcCiQF2ejRH1HYSDOIRPYtg4JNqa65xS3ioAK2MR0SpdKbvMIPyW2NRFoAVDlAfrKcT6kcTmklUmpY89kUgQVn3w8f04gfdzNGLUJnzGQF1SKpNRi25jN.png" width="600">

NOTICE: if you are going to reproduce [this notebook](https://github.com/supervisely/supervisely/blob/master/help/jupyterlab_scripts/src/tutorials/09_detection_segmentation_pipeline/detection_segmentation_pipeline.ipynb), just copy/paste inference modes from there. 

### Python example with no connection to Supervisely SDK

Let's repeat the procedure we've just did with CURL and sliding window mode. Here is the python script that does the job.

```py
import requests
from requests_toolbelt import MultipartEncoder


if __name__ == '__main__':
    content_dict = {}
    content_dict['image'] = ("big_image.png", open("/workdir/src/big_image.jpg", 'rb'), 'image/*')
    content_dict['mode'] = ("mode", open('/workdir/src/sliding_window_mode_example.json', 'rb'))

    encoder = MultipartEncoder(fields=content_dict)
    response = requests.post("http://0.0.0.0:5000/model/inference", data=encoder, headers={'Content-Type': encoder.content_type})
    print(response.json())
```


### How to place model weights inside Docker image

Just create a Dockerfile in the same directory with NN weights. Something like this:

```
.
├── Dockerfile
└── model
    ├── config.json
    └── model.pt
```

Dockerfile content:
```
FROM supervisely/nn-yolo-v3
COPY model /sly_task_data/model
```

To build image just execute the following command in the Dockerfile's directory:
```sh 
docker build -t my_super_image_with_model_inside .
```
To check that the model is inside our new image ```my_super_image_with_model_inside``` run this command:

```sh
docker run --rm -it my_super_image_with_model_inside bash -c 'ls -l /sly_task_data/model'
```

The command output should look like this:
```
otal 238432
-rw-r--r-- 1 root root       870 Feb 17 11:57 config.json
-rw-r--r-- 1 root root 244142107 Feb 17 11:57 model.pt
```

REMINDER: Once you place a model inside docker image you do not need to mount model weights, i.e. ```docker run``` command will be:

```sh
docker run --rm -it \
    --runtime=nvidia \
    -p 5000:5000 \
    --env GPU_DEVICE=0 \
    my_super_image_with_model_inside \
    python /workdir/src/rest_inference.py
```

# Done!
