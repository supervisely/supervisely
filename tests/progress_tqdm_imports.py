# case1 # monkeypatchin not working in venv
from tqdm import tqdm  # isort: skip
import supervisely as sly  # isort: skip

# case2 #works ok everywhere
# import supervisely as sly  # isort: skip
# from tqdm import tqdm  # isort: skip

# case3
# import tqdm  # isort: skip
# import supervisely as sly  # isort: skip

# case4
# import supervisely as sly  # isort: skip
# import tqdm  # isort: skip

import os
import shutil

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()


# breakpoint()
def upl():
    size = sly.fs.get_file_size("/home/grokhi/Downloads/google-chrome-stable_current_amd64.deb")
    progress = tqdm(desc="Uploading", total=size, unit_scale=True, unit="B", miniters=2, position=1)
    # progress = tqdm.tqdm(
    #     desc="Uploading", total=size, unit_scale=True, unit="B", miniters=2, position=1
    # )  # case 3
    api.file.upload(
        449,
        "/home/grokhi/Downloads/google-chrome-stable_current_amd64.deb",
        "google-chrome-stable_current_amd64.deb",
        progress,
    )
    print("1 (positional, non-idempotent args)")


# dwnl()
upl()
# dwnl_dir()
# upldir()

breakpoint()
TEAM_ID = 449
TF_FILEPATH = "/google-chrome-stable_current_amd64.deb"


def dwnl():
    p = tqdm(
        desc=f"download",
        total=api.file.get_directory_size(TEAM_ID, TF_FILEPATH),
        unit="B",
        unit_scale=True,
    )
    api.file.download(
        449, TF_FILEPATH, "/tmp/google-chrome-stable_current_amd64.deb", progress_cb=p
    )
    print("2 (pos + keyword)")


TF_DIRPATH = "/stats/"
LOC_DIRPATH = "/tmp/stats/"


def dwnl_dir():
    p = tqdm(
        desc=f"download_directory",
        unit="B",
        unit_scale=True,
    )
    p.total = api.file.get_directory_size(TEAM_ID, TF_DIRPATH)
    api.file.download_directory(TEAM_ID, TF_DIRPATH, LOC_DIRPATH, progress_cb=p)
    print("2.1 total after __init__")


def upldir():
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


def dwnl_prj():
    project = api.project.get_info_by_id(32796)
    p = tqdm(
        desc="download",
        total=project.items_count,
    )
    sly.download(api, project.id, "/tmp/lemons/", progress_cb=p)
    print("3")


def upl_prj():
    project_fs = sly.read_project("/tmp/lemons/")
    p = tqdm(
        desc="upload",
        total=project_fs.total_items,
    )
    sly.upload("/tmp/lemons/", api, 691, progress_cb=p)
    print("4")

    shutil.rmtree("/tmp/lemons/")
    os.makedirs("/tmp/lemons/", exist_ok=True)


def dwn_prj_img():
    project = api.project.get_info_by_id(32796)
    p = tqdm(
        desc="download",
        total=project.items_count,
    )
    sly.download_project(api, project.id, "/tmp/lemons/", progress_cb=p)
    print("5")


def upl_prj_img():
    project_fs = sly.read_project("/tmp/lemons/")
    p = tqdm(
        desc="upload",
        total=project_fs.total_items,
    )
    sly.upload_project("/tmp/lemons/", api, 691, progress_cb=p)
    print("6")


def dwnl_prj_vid():
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


def dwnl_prj_vol():
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


def dwnl_prj_pcl():
    project = api.project.get_info_by_id(18592)
    p = tqdm(
        desc="download",
        total=project.items_count,
    )
    sly.download_pointcloud_project(api, project.id, "/tmp/pcl/", progress_cb=p)
    print("11")


def dwnl_prj_pclep():
    project = api.project.get_info_by_id(18593)
    p = tqdm(
        desc="download",
        total=project.items_count,
    )
    sly.download_pointcloud_episode_project(api, project.id, "/tmp/pcl_ep/", progress_cb=p)
    print("12")
