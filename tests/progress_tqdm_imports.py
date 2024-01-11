# case1
from tqdm import tqdm  # isort: skip
import supervisely as sly  # isort: skip

# case2 #works ok
# import supervisely as sly  # isort: skip
# from tqdm import tqdm  # isort: skip

import os
import shutil

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

##################################
# api.file DOWNLOAD/UPLOAD
##################################
size = sly.fs.get_file_size("/home/grokhi/Downloads/google-chrome-stable_current_amd64.deb")
progress = tqdm(desc="Uploading", total=size, unit_scale=True, unit="B")
api.file.upload(
    449,
    "/home/grokhi/Downloads/google-chrome-stable_current_amd64.deb",
    "google-chrome-stable_current_amd64.deb",
    progress_cb=progress,
)
print("1")

TEAM_ID = 449
TF_FILEPATH = "/google-chrome-stable_current_amd64.deb"
p = tqdm(
    desc=f"download",
    total=api.file.get_directory_size(TEAM_ID, TF_FILEPATH),
    unit="B",
    unit_scale=True,
)
api.file.download(449, TF_FILEPATH, "/tmp/google-chrome-stable_current_amd64.deb", progress_cb=p)
print("2")

##################################
# sly DOWNLOAD/UPLOAD
##################################
breakpoint()
project = api.project.get_info_by_id(32796)
p = tqdm(
    desc="download",
    total=project.items_count,
)
sly.download(api, project.id, "/tmp/lemons/", progress_cb=p)
print("3")

# breakpoint()
project_fs = sly.read_project("/tmp/lemons/")
p = tqdm(
    desc="upload",
    total=project.items_count,
)
# print(p)
sly.upload("/tmp/lemons/", api, 691, progress_cb=p)
print("4")

shutil.rmtree("/tmp/lemons/")

breakpoint()
project_fs = sly.read_project("/tmp/lemons/")
p = tqdm(
    desc="upload",
    total=project.items_count,
)
# print(p)
sly.upload("/tmp/lemons/", api, 691, progress_cb=p)
print("5")
