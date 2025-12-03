import os
from datetime import datetime

from supervisely import Api
from supervisely.project.volume_project import VolumeProject
from supervisely.volume import VolumeDataVersion

api = Api.from_env()
ver = VolumeDataVersion(api)

TEAM_ID = 9
PROJECT_ID = 2451
# check if api method works
# ver_id, ver_token = ver.reserve(PROJECT_ID)

res = VolumeProject.download_bin(api, PROJECT_ID, "local_volume_project")
# new_projinfo = VolumeProject.upload_bin(api, res, 66, "uploaded_volume_project_8")

volume_version = VolumeDataVersion(api)
volume_version.project_info = api.project.get_info_by_id(PROJECT_ID)

# timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# version_dir = f"/system/versions/{PROJECT_ID}"
# # if not api.file.dir_exists(TEAM_ID, version_dir):
# #     api.file.mkdir(TEAM_ID, version_dir)
# path = os.path.join(version_dir, timestamp + ".tar.zst")
# fileinfo = volume_version._compress_and_upload(path)

backup_files = "/system/versions/2451/20251203053021.tar.zst"
bin_io = volume_version._download_and_extract(backup_files)
new_project_info = volume_version.project_cls.upload_bin(
    api,
    bin_io,
    8,
    skip_missed=True,
)

ver.project_info = api.project.get_info_by_id(PROJECT_ID)
print("")


version_list = ver.get_list(PROJECT_ID)
print("Existing versions:", version_list)

map = ver.get_map(PROJECT_ID)
print("Version map:", map)

version_id = ver.create(PROJECT_ID, "test version", "first version test")
if version_id is not None:
    print("Created version ID:", version_id)
else:
    print("Version creation failed.")

if version_id is not None:
    ver.restore(PROJECT_ID, version_id=version_id)
