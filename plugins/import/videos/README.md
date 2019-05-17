# Import Videos 
This plugin allows you to upload videos by cutting them into frames with a given step. It can work with one or few video files. The name of the video file corresponds to the name of the resulting dataset. Supported video formats - **"avi"** and **"mp4"**.

File structure which you can drag and drop should look like this:

```
.
├── video_01.mp4
├── video_02.mp4
└── video_03.avi
```

### Settings config

```json
{
    "step": 25,
    "start_frame": 0,
    "end_frame": -1,
    "skip_frames": [],
    "dhash_min_hamming_distance": 0
}
```

Configuration options:
* `step`: how many frames to skip before considering a frame for import.
    Setting `step` to 1 will import every frame. Setting `step` to 25 (the
    default setting) will import every 25-th frame.
* `start_frame`: skip all the frames until this frame number is reached in the
    video. Frame indexing is zero based, so the first frame of the video has index 0.
* `end_frame`: skip all the frames that come after this frame index in the video.
* `skip_frames`: skip the frame indices from this list.
* `dhash_min_hamming_distance`: minimum [dHash](https://pypi.org/project/dhash)
    Hamming distance from the previously imported frame. This setting allows one
    to skip frames that are too similar to the previously imported frames.
    Higher values enforce larger difference.


### Example
In this example, we will upload one video with “skip frame” value set to 60 which means that every 2 seconds we will extract a frame (under the conditions that a video recorded with 30 fps).
![](https://i.imgur.com/c4BvQJO.gif)
