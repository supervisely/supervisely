import os

from dotenv import load_dotenv
import supervisely as sly
from supervisely.video_annotation.video_tag import VideoTag
from supervisely.video_annotation.video_tag_collection import VideoTagCollection

# Init api for communicating with Supervisely Instance
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api.from_env()

project_id = 15626
dataset_id = 53496
video_id = 17525980
video_ids = [17530148, 17525985, 17530150]

# Get variables from environment
# project_id = int(os.environ["modal.state.slyProjectId"])
# dataset_id = int(os.environ["modal.state.slyDatasetId"])
# video_id = int(os.environ["modal.state.slyVideoId"])


# Get video info by id
video_info = api.video.get_info_by_id(video_id)
print(video_info)


# Get a list of all videos in the dataset
video_info_list = api.video.get_list(dataset_id)
print(video_info_list)


# Download the video by id to the local path
dir_path = "/home/admin/work/projects/videos/"
save_path = os.path.join(dir_path, video_info.name)
api.video.download_path(video_id, save_path)


# Download the frame of the video
frame_idx = 15
file_name = "frame.png"
save_path = os.path.join(dir_path, file_name)
api.video.frame.download_path(video_id, frame_idx, save_path)


# Download the range of video frames as images
start_fr = 5  # start frame
end_fr = 8  # end frame
frame_indexes = [i for i in range(start_fr, end_fr)]
save_paths = [os.path.join(dir_path, f"{idx}.png") for idx in frame_indexes]
api.video.frame.download_paths(video_id, frame_indexes, save_paths)

# Upload the video from the directory
local_path = os.path.join(dir_path, "video.mp4")
upload_info = api.video.upload_path(dataset_id=dataset_id, name="My_video", path=local_path)
print(upload_info)


# Upload Video from given hash to Dataset.
hash = video_info.hash
name = "My new video2"
new_video_info = api.video.upload_hash(dataset_id, name, hash)
print(new_video_info)


# Upload a list of videos from the directory
local_path1 = "/home/admin/work/projects/videos/myvideo1.mp4"
local_path2 = "/home/admin/work/projects/videos/myvideo2.mp4"

upload_info = api.video.upload_paths(
    dataset_id=dataset_id,
    names=["video1.mp4", "video2.mp4"],
    paths=[local_path1, local_path2],
)
print(upload_info)


# Upload a list of videos by hashes
src_dataset_id = 53496
dst_dataset_id = 53497
hashes = []
names = []
metas = []
video_info_list = api.video.get_list(src_dataset_id)
# Create lists of hashes, videos names and meta information for each video
for video_info in video_info_list:
    hashes.append(video_info.hash)
    # It is necessary to upload videos with the same names(extentions) as in src dataset
    names.append(video_info.name)
    metas.append({video_info.name: video_info.frame_height})
new_videos_info = api.video.upload_hashes(dst_dataset_id, names, hashes, metas)
print(new_videos_info)


# Download annotations for the single video by id
ann = api.video.annotation.download(video_id)

print(ann)


# Download annotations for a list of the videos by ids

anns = api.video.annotation.download_bulk(dataset_id, video_ids)


#  Work with annotations and tags
height, width = 500, 700
frames_count = 1

# VideoObjectCollection
obj_class_car = sly.ObjClass("car", sly.Rectangle)
video_obj_car = sly.VideoObject(obj_class_car)
objects = sly.VideoObjectCollection([video_obj_car])

# FrameCollection
fr_index = 7
geometry = sly.Rectangle(0, 0, 100, 100)
video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
frame = sly.Frame(fr_index, figures=[video_figure_car])
frames = sly.FrameCollection([frame])

# VideoTagCollection
meta_car = sly.TagMeta("car_tag", sly.TagValueType.ANY_STRING)
vid_tag = VideoTag(meta_car, value="acura")
video_tags = VideoTagCollection([vid_tag])

# Description
descr = "car example"
video_ann = sly.VideoAnnotation((height, width), frames_count, objects, frames, video_tags, descr)


# Add metadata to the video
local_path = "/home/admin/work/projects/videos/video.mp4"
upload_info = api.video.upload_path(
    dataset_id=dataset_id, name="My_new_video", path=local_path, meta={1: "meta_example"}
)
print(upload_info)


dst_dataset_id = 53497
hashes = [video_info.hash for video_info in video_info_list]
names = [video_info.name for video_info in video_info_list]
metas = [{video_info.name: "meta_example"} for video_info in video_info_list]
new_videos_info = api.video.upload_hashes(dst_dataset_id, names, hashes, metas)
print(new_videos_info)
