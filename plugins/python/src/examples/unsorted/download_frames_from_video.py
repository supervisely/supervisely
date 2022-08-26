import os
import supervisely as sly

os.environ['SERVER_ADDRESS'] = "https://app.supervise.ly/"
os.environ['API_TOKEN'] = "YOUR API TOKEN"
api = sly.Api.from_env()

video_id = 2859430
video_info = api.video.get_info_by_id(video_id)

all_frame_indexes = list(range(video_info.frames_count))
specific_frame_indexes = [0, 1, 2, 50, 15, 3, 120]

# download frames as numpy ndarrays
frames = api.video.frame.download_nps(video_id=video_id, frame_indexes=specific_frame_indexes)

# download frames and save them to given paths
save_dir = "/home/admin/projects/video_project/frames/"
save_paths = [f"{save_dir}{idx}.png" for idx in specific_frame_indexes]
api.video.frame.download_paths(video_id=video_id, frame_indexes=specific_frame_indexes, paths=save_paths)
