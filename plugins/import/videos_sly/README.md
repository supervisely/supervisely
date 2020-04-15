# Import Videos with annotations in Supervisely format 
This plugin allows you to upload Video Projects in Supervisely format which includes videos, annotations and `meta.json`. More about Supervisely format you can read [here](https://docs.supervise.ly/ann_format/).

**IMPORTANT NOTICE**: 

Supervisely Community Edition (CE) has some limitations on video file size, also it is only possible to upload video files in `[mp4, webm, ogg, ogv]` containers with the following codecs `[h264, vp8, vp9]`. 

Enterprise Edition (EE) DON'T have limitations:
- files in most popular formats can be uploaded: `[avi, mp4, 3gp, flv, webm, wmv, mov, mkv, ...]`
- it supports realtime video streaming and transcoding on the fly, thus no need to convert and store data in different formats
- it has built-in tools to connect existing video storages: add your private local storage or the cloud one (Google Cloud, Amazon AWS, Microsoft Azure) 

#Example:

For this format the structure of directory should be the following:

```
my_project
├── meta.json
├── dataset_name_01
│   ├── ann
│   │   ├── video1.mp4.json
│   │   ├── video2.avi.json
│   │   └── video3.mov.json
│   └── video
│       ├── video1.mp4
│       ├── video2.avi
│       └── video3.mov
├── dataset_name_02
│   ├── ann
│   │   ├── video1.mp4.json
│   │   ├── video2.avi.json
│   │   └── video3.mov.json
│   └── video
│       ├── video1.mp4
│       ├── video2.avi
│       └── video3.mov
```

Directory "my_project" contains two folders and file `meta.json`. For each folder will be created corresponding dataset inside project. As you can see, videos are separated from annotations.


