import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

local_path = "/Users/max/Downloads/file_example_MP4_480_1_5MG.mp4"

dataset_id = 1016
name = sly.fs.get_file_name_with_ext(local_path)

project = api.video.upload_path(
    dataset_id=dataset_id, name=name, path=local_path, item_progress=True
)
