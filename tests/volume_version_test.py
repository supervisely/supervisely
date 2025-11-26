from supervisely import Api
from supervisely.project.volume_project import VolumeProject
from supervisely.volume import VolumeDataVersion

api = Api.from_env()
ver = VolumeDataVersion(api)

PROJECT_ID = 2452

res = VolumeProject.download_bin(api, PROJECT_ID, "local_volume_project")

ver.project_info = api.project.get_info_by_id(PROJECT_ID)
anns = ver._collect_annotation_blobs()
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