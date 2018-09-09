# Description
This repo contains a Jupyter Notebook that makes inference of unet model.

# Clone repository
``` 
git clone https://github.com/supervisely/supervisely.git
```

# Preparation with NN weights
Download NN from your account. Then unpack archive to the folder `tutorials/02_unet_inference/data/model`. For example, `02_unet_inference` folder will look like this:

```
.
├── data
│   ├── img
│   │   └── 00000220.png
│   └── model
│       ├── config.json
│       └── model.pt
├── docker
│   ├── Dockerfile
│   └── run.sh
├── README.md
├── result.png
└── src
    └── 02_unet_inference.ipynb

```

# How to run
Execute the following commands:

```
cd tutorials/02_unet_inference/docker
./run.sh
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

After running `02_unet_inference.ipynb`, you get the following results:
![Drivable area segmentation](result.png)

