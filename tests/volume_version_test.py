from supervisely import Api
from supervisely.project import DataVersion

api = Api.from_env()

TEAM_ID = 9
PROJECT_ID = 5064

ver = DataVersion(api)

mapping = ver.create(PROJECT_ID, "new volume version from test")
print("Created mapping:", mapping)
