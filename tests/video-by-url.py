import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

project_id = 13392
dataset_id = 50871

# video_link = "https://download.samplelib.com/mp4/sample-5s.mp4"
# res = api.video.upload_link(dataset_id, link=video_link)
# print(res[0])

video_id = 3353173
info = api.video.get_info_by_id(video_id)
info2 = api.video.add_existing(dataset_id, info, name="abcd.mp4")
print(info2.name)
