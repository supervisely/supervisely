import requests
import os
import json
import pandas as pd
import supervisely_lib as sly

WORKSPACE_ID = int('%%WORKSPACE_ID%%')
src_project_name = '%%IN_PROJECT_NAME%%'

api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

project = api.project.get_info_by_name(WORKSPACE_ID, src_project_name)
if project is None:
    raise RuntimeError('Project {!r} not found'.format(src_project_name))

# implementaiton without supervisely_python sdk
url = os.environ['SERVER_ADDRESS']+'public/api/v3/projects.activity'
payload = {'id': project.id}
r = requests.post(url, json=payload, headers={'Content-Type': 'application/json', 'x-api-key': os.environ['API_TOKEN']})
df = pd.DataFrame(json.loads(r.text))

#or alternative one line implementation
#df = api.project.get_activity(project.id)

dest_dir = os.path.join(sly.TaskPaths.OUT_ARTIFACTS_DIR, src_project_name)
sly.fs.mkdir(dest_dir)
df.to_csv(os.path.join(dest_dir, 'activity.csv'))

sly.logger.info('Activity saved to activity.csv file')