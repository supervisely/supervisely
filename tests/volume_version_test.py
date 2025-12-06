import os
from datetime import datetime

import supervisely as sly
from supervisely import Api
from supervisely.project import DataVersion
from supervisely.project.volume_project import VolumeProject

api = Api.from_env()

TEAM_ID = 9
PROJECT_ID = 5032
# check if api method works
# ver_id, ver_token = ver.reserve(PROJECT_ID)

# res = VolumeProject.download_bin(api, PROJECT_ID, "local_volume_project")
# print("Downloaded to:", res)
# new_projinfo = VolumeProject.upload_bin(api, res, 8, "uploaded_volume_project_8")
# print("Uploaded project id:", new_projinfo.id)

ver = DataVersion(api)

new_projectinfo = ver.restore(5032, version_num=1)
print("Restored project id:", new_projectinfo.id)
