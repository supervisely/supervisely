# Import Videos

Just drag and drop the collection of video files and folders with them. The new project will be created. 

Download [`videos.tar`](https://drive.google.com/uc?id=1JAmWva6NyPj1-TYeAyCbJSxYK_MIRzg8&export=download) example, unpack archive and drag-and-drop its content.  


**IMPORTANT NOTICE**: 

Supervisely Community Edition (CE) has some limitations on video file size, also it is only possible to upload video files in `[mp4, webm, ogg, ogv]` containers with the following codecs `[h264, vp8, vp9]`. 

Enterprise Edition (EE) DON'T have limitations:
- files in most popular formats can be uploaded: `[avi, mp4, 3gp, flv, webm, wmv, mov, mkv, ...]`
- it supports realtime video streaming and transcoding on the fly, thus no need to convert and store data in different formats
- it has built-in tools to connect existing video storages: add your private local storage or the cloud one (Google Cloud, Amazon AWS, Microsoft Azure)   


## Datasets structure

Plugin creates datasets with names of top-most directories in a hierarchy. Files from root import directory will be placed to dataset with name "ds0".  

Let's consider several examples.
 
Example 1. Import structure:

```
.
├── cooking.mp4
├── dataset_animals
│   ├── sea_lion.mp4
│   └── zebras.mp4
└── dataset_cars
    ├── cars.mp4
    └── cross_roads.mp4
```

In this case the following datasets will be created

- `ds_0` with file `cooking.mp4`
- `dataset_animals` with two files `sea_lion.mp4` and `zebras.mp4`
- `dataset_cars` with two files `cars.mp4` and `cross_roads.mp4`

![example 1](https://i.imgur.com/15J67BG.png)

Example 2. Import structure:

```
abcd_folder/
├── cooking.mp4
├── dataset_animals
│   ├── sea_lion.mp4
│   └── zebras.mp4
└── dataset_cars
    ├── cars.mp4
    └── cross_roads.mp4
```

In this case only the one dataset `abcd_folder` will be created with all video files.


![example 2](https://i.imgur.com/s3AhlZ9.png)