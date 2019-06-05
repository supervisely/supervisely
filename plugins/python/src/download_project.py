import os
import supervisely_lib as sly

WORKSPACE_ID = %%WORKSPACE_ID%%
src_project_name = '%%IN_PROJECT_NAME%%'

api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

#### End settings. ####

sly.logger.info('DOWNLOAD_PROJECT', extra={'title': src_project_name})
project = api.project.get_info_by_name(WORKSPACE_ID, src_project_name)
sly.download_project(
    api, project.id, os.path.join(sly.TaskPaths.OUT_ARTIFACTS_DIR, src_project_name), log_progress=True)
sly.logger.info('Project {!r} has been successfully downloaded.'.format(src_project_name))
