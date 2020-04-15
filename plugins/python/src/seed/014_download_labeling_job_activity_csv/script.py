import os
import supervisely_lib as sly

job_id = int('%%JOB_ID%%')

api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

job_info = api.labeling_job.get_info_by_id(job_id)
if job_info is None:
    raise RuntimeError('Labeling job id={!r} not found'.format(job_id))

df = api.labeling_job.get_activity(job_id)

dest_dir = os.path.join(sly.TaskPaths.OUT_ARTIFACTS_DIR, "job_id_{}_name_{}".format(job_info.id, job_info.name))
sly.fs.mkdir(dest_dir)

df.to_csv(os.path.join(dest_dir, 'activity.csv'))

sly.logger.info('Activity saved to activity.csv file')