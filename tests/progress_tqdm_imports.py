# case1 # monkeypatchin not working in venv
from tqdm import tqdm  # isort: skip
import supervisely as sly  # isort: skip

# case2 #works ok everywhere
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
progress = tqdm(desc="Uploading", total=size, unit_scale=True, unit="B", miniters=2, position=1)
api.file.upload(
    449,
    "/home/grokhi/Downloads/google-chrome-stable_current_amd64.deb",
    "google-chrome-stable_current_amd64.deb",
    progress,
)
print("1 (positional, non-idempotent args)")

TEAM_ID = 449
TF_FILEPATH = "/google-chrome-stable_current_amd64.deb"
p = tqdm(
    desc=f"download",
    total=api.file.get_directory_size(TEAM_ID, TF_FILEPATH),
    unit="B",
    unit_scale=True,
)
api.file.download(449, TF_FILEPATH, "/tmp/google-chrome-stable_current_amd64.deb", progress_cb=p)
print("2 (pos + keyword)")

breakpoint()

TF_DIRPATH = "/stats/"
LOC_DIRPATH = "/tmp/stats/"
p = tqdm(
    desc=f"download_directory",
    unit="B",
    unit_scale=True,
)
p.total = api.file.get_directory_size(TEAM_ID, TF_DIRPATH)
api.file.download_directory(TEAM_ID, TF_DIRPATH, LOC_DIRPATH, progress_cb=p)
print("2.1 total after __init__")

p = tqdm(
    desc=f"upload_directory",
    total=sly.fs.get_directory_size(LOC_DIRPATH),
    unit="B",
    unit_scale=True,
)
api.file.upload_directory(
    TEAM_ID,
    LOC_DIRPATH,
    TF_DIRPATH,
    progress_size_cb=p,
)
print("2.2")

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

project_fs = sly.read_project("/tmp/lemons/")
p = tqdm(
    desc="upload",
    total=project.items_count,
)
sly.upload("/tmp/lemons/", api, 691, progress_cb=p)
print("4")

shutil.rmtree("/tmp/lemons/")
os.makedirs("/tmp/lemons/", exist_ok=True)

breakpoint()
project = api.project.get_info_by_id(32796)
p = tqdm(
    desc="download",
    total=project.items_count,
)
sly.download_project(api, project.id, "/tmp/lemons/", progress_cb=p)
print("5")

project_fs = sly.read_project("/tmp/lemons/")
p = tqdm(
    desc="upload",
    total=project.items_count,
)
sly.upload_project("/tmp/lemons/", api, 691, progress_cb=p)
print("6")

breakpoint()
project = api.project.get_info_by_id(18142)
p = tqdm(
    desc="download",
    total=project.items_count,
)
sly.download_video_project(api, project.id, "/tmp/vid/", progress_cb=p)
print("7")

# #! no progress_cb in sly.upload_video_project
# project_fs = sly.read_project("/tmp/vid/")
# p = tqdm(
#     desc="upload",
#     total=project.items_count,
# )
# sly.upload_video_project("/tmp/vid/", api, 691, progress_cb=p)
# print("8")

breakpoint()
project = api.project.get_info_by_id(18594)
p = tqdm(
    desc="download",
    total=project.items_count,
)
sly.download_volume_project(api, project.id, "/tmp/vol/", progress_cb=p)
print("9")
# #! no progress_cb in sly.upload_volume_project
# project_fs = sly.read_project("/tmp/vol/")
# p = tqdm(
#     desc="upload",
#     total=project.items_count,
# )
# sly.upload_volume_project("/tmp/vol/", api, 691, progress_cb=p)
# print("10")

breakpoint()
project = api.project.get_info_by_id(18592)
p = tqdm(
    desc="download",
    total=project.items_count,
)
sly.download_pointcloud_project(api, project.id, "/tmp/pcl/", progress_cb=p)
print("11")

breakpoint()
project = api.project.get_info_by_id(18593)
p = tqdm(
    desc="download",
    total=project.items_count,
)
sly.download_pointcloud_episode_project(api, project.id, "/tmp/pcl_ep/", progress_cb=p)
print("12")


# api.annotation.download_batch(59589, [19425323, 19425319], progress_cb=p)
# api.annotation.download_json_batch(59589, [19425323, 19425319], progress_cb=p)

# api.annotation.upload_anns  # progress_cb(len(batch))
# api.annotation.upload_jsons  # progress_cb(len(batch))
# api.annotation.upload_paths  # progress_cb(len(batch))

# api.app.upload_dtl_archive # progress_cb(read_mb)
# api.app.upload_files # Method is unavailable

# api.image.download_bytes(61226, [19489032, 19489521, 19489360], progress_cb=p)

# api.image.download_nps  # progress_cb(1)
# api.image.download_paths(61226, image_ids, save_paths, progress_cb=p)
# api.image.download_paths  # progress_cb(1)
# api.image.download_paths_by_hashes  # progress_cb(1)
# api.image.upload_hashes  # progress_cb(len(images))
# api.image.upload_ids # progress_cb(len(images)),
# api.image.upload_nps # progress_cb(len(remote_hashes)), progress_cb(len(hashes_rcv))
# api.image.upload_paths # progress_cb(len(remote_hashes)), progress_cb(len(hashes_rcv))
# api.model.remove_batch  # progress_cb(1)
# api.model.upload  # progress_cb(read_mb)
# api.model.download_to_dir  # progress_cb(read_mb)
# api.model.download_to_tar  # progress_cb(read_mb)
# api.task.upload_dtl_archive  # progress_cb(read_mb)
# api.task.upload_files  # progress_cb(len(remote_hashes)), progress_cb(len(content_dict))


# api.video.download_path  # progress_cb(len(chunk))
# video_infos = api.video.upload_paths(
#     dataset_id=dataset_id, names=video_names, paths=video_paths, progress_cb=p  # .iters_done_report
# )
# api.video.upload_paths()  # progress_cb(len(remote_hashes)), progress_cb(len(hashes_rcv))
# new_volumes_info = api.volume.upload_hashes(
# api.volume.download_path(volume_info.id, path, progress_cb=p)
