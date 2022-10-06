import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

# optimized version
src_path = "agent://777/coco128/README.txt"
dst_path = "/Users/max/work/app_debug_data/optimized_coco128_README.txt"

if sly.fs.file_exists(dst_path):
    sly.fs.silent_remove(dst_path)
sly.fs.ensure_base_path(dst_path)
api.file.download(team_id=7, remote_path=src_path, local_save_path=dst_path)
