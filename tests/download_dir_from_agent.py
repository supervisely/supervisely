import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

# from agent
# src_path = "agent://1/[dev] Names generator/13233/"
# dst_path = "/Users/max/work/app_debug_data/dir_from_agent"

# from team files
# src_path = "/unet_01/"
# dst_path = "/Users/max/work/app_debug_data/dir_from_team_files"

# optimized version
src_path = "agent://777/coco128"
dst_path = "/Users/max/work/app_debug_data/optimized_coco128"

if sly.fs.dir_exists(dst_path):
    sly.fs.remove_dir(dst_path)
sly.fs.ensure_base_path(dst_path)
api.file.download_directory(team_id=7, remote_path=src_path, local_save_path=dst_path)
