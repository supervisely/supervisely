from tqdm import tqdm  # isort: skip
import os

from dotenv import load_dotenv

import supervisely as sly

load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()
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

breakpoint()
bproject = api.project.get_info_by_id(32796)
# print(p)
p = tqdm(
    desc="api.file.download",
    total=project.items_count,
    is_size=False,
)
sly.download(api, project.id, "/tmp/lemons/", progress_cb=p)
print("3")

project_fs = sly.read_project("/tmp/lemons/")
p = tqdm(
    desc="api.file.upload",
    total=project.items_count,
    is_size=False,
)
# print(p)
sly.upload("/tmp/lemons/", api, 691)
print("4")
