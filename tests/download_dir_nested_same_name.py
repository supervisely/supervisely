import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

src_path = "/val/"
dst_path = "/Users/max/work/app_debug_data/val/"

if sly.fs.dir_exists(dst_path):
    sly.fs.remove_dir(dst_path)
sly.fs.ensure_base_path(dst_path)
api.file.download_directory(team_id=7, remote_path=src_path, local_save_path=dst_path)
