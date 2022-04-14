import supervisely

# define variables
AGENT_ID = 49
TEAM_ID = 305
WORKSPACE_ID = 437
APP_NAMES_TO_LAUNCH = tuple(['RITM interactive segmentation SmartTool'])

api = supervisely.Api.from_env()

# get applications ids to start
apps_list_in_team = api.app.get_list(team_id=TEAM_ID)
apps_list_to_launch = [app_info for app_info in apps_list_in_team if app_info['name'] in APP_NAMES_TO_LAUNCH]

supervisely.logger.info(f'{apps_list_to_launch=}')

# launch applications in cycle
launched_tasks_ids = []

for current_app in apps_list_to_launch:
    app_default_state = api.app.get_info_by_id(id=current_app['id'])['config'].get('modalTemplateState', {})
    app_default_state.update({'device': 'cpu'})  # update by custom params

    launched_tasks_list = api.task.start(
        agent_id=AGENT_ID,
        app_id=current_app['id'],
        workspace_id=WORKSPACE_ID,
        params={'state': app_default_state}
    )

    launched_tasks_ids.extend(list(map(lambda task_elem: task_elem['taskId'], launched_tasks_list)))  # extract task_ids
    supervisely.logger.info(f'application {current_app["name"]=}, {current_app["id"]=} launched')

# show launched applications statuses
supervisely.logger.info(f'{len(launched_tasks_ids)} task(-s) launched')
supervisely.logger.info(f'{launched_tasks_ids=}')

for task_id in launched_tasks_ids:
    task_info = api.task.get_info_by_id(id=task_id)
    supervisely.logger.info(f'{task_info["status"]=}')

