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
