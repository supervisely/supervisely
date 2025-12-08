from supervisely import Api
from supervisely.project import DataVersion

api = Api.from_env()

TEAM_ID = 9
PROJECT_ID = 2444

ver = DataVersion(api)

version = ver.create(PROJECT_ID, "new volume version from test1")
print("Created version:", version)

new_projinfo = ver.restore(PROJECT_ID, version)
print("Restored to version, new project id:", new_projinfo.id)
