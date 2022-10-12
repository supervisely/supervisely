import os
from dotenv import load_dotenv
import supervisely as sly


load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()

# path in Team Files
src_remote_path = "/my-folder/my_model.h5"

# local path
dst_local_path = "/Users/max/Downloads/abc/my_model.h5"

# ensure that directory structure (/Users/max/Downloads/abc/) exists
sly.fs.ensure_base_path(dst_local_path)

# download file from Supervisely Team Files to local computer
api.file.download(team_id=7, remote_path=src_remote_path, local_save_path=dst_local_path)

if sly.fs.file_exists(dst_local_path):
    print("file has been successfully downloaded from Supervisely Team Files")
