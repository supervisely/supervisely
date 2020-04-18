import os
import supervisely_lib as sly

WORKSPACE_ID = int('%%WORKSPACE_ID%%')
src_project_name = '%%IN_VIDEO_PROJECT_NAME%%'
src_dataset_ids = %%DATASET_IDS:None%%

# for debug
# WORKSPACE_ID = 8
# src_project_name = 'videos'
# src_dataset_ids = None
# download_videos = False


api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

#### End settings. ####

project = api.project.get_info_by_name(WORKSPACE_ID, src_project_name)
if project is None:
    raise RuntimeError('Project {!r} not found'.format(src_project_name))

if project.type != str(sly.ProjectType.VIDEOS):
    raise RuntimeError('Project {!r} has type {!r}. This script works only with video projects'
                       .format(src_project_name, project.type))

dest_dir = os.path.join(sly.TaskPaths.OUT_ARTIFACTS_DIR, src_project_name)
sly.fs.mkdir(dest_dir)

sly.download_video_project(api, project.id, dest_dir, src_dataset_ids, download_videos=False, log_progress=True)
sly.logger.info('Project {!r} has been successfully downloaded'.format(src_project_name))
