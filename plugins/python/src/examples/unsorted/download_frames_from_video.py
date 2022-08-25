import os
import supervisely as sly

os.environ['SERVER_ADDRESS'] = "https://app.supervise.ly/"
os.environ['API_TOKEN'] = "YOUR API TOKEN"
api = sly.Api.from_env()

video_id = 2859430
video_info = api.video.get_info_by_id(video_id)

all_frame_ids = list(range(video_info.frames_count))
specific_frame_ids = [0, 1, 2, 50, 15, 3, 120]

# download frames as numpy ndarrays
frames = api.video.frame.download_nps(video_id=video_id, ids=specific_frame_ids)

# download frames and save them to given paths
save_paths = [f"/home/admin/projects/video_project/frames/{idx}.png" for idx in specific_frame_ids]
api.video.frame.download_paths(video_id=video_id, ids=specific_frame_ids, paths=save_paths)
