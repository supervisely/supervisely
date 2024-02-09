# case1 # monkeypatchin not working in venv
from tqdm import tqdm  # isort: skip
import supervisely as sly  # isort: skip

# case2 #works ok everywhere
# import supervisely as sly  # isort: skip
# from tqdm import tqdm  # isort: skip

# case3
# import tqdm  # isort: skip
# import supervisely as sly  # isort: skip

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
