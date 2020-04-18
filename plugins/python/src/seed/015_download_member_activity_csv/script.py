import os
import supervisely_lib as sly

team_id = int('%%TEAM_ID%%')
user_id = int('%%USER_ID%%')

api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

team_info = api.team.get_info_by_id(team_id)
if team_info is None:
    raise RuntimeError('Team id={!r} not found'.format(team_id))

df = api.user.get_member_activity(team_id, user_id)

dest_dir = os.path.join(sly.TaskPaths.OUT_ARTIFACTS_DIR, "user_{}".format(user_id))
sly.fs.mkdir(dest_dir)

df.to_csv(os.path.join(dest_dir, 'activity.csv'))

sly.logger.info('Activity saved to activity.csv file')