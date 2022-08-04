import supervisely


"""
BE SURE THAT YOU HAVE ENV VARIABLES

SERVER_ADDRESS="https://app.supervise.ly/"  # or your instance address
API_TOKEN=""                                # get it in https://app.supervise.ly/user/settings/tokens
AGENT_TOKEN=""                              # get it in https://app.supervise.ly/nodes/list
"""


WORKSPACE_ID = 437

api = supervisely.Api.from_env()

all_tasks_info = api.task.get_list(workspace_id=WORKSPACE_ID, filters=[{'field': 'status',
                                                                        'operator': '=',
                                                                        'value': api.task.Status.STARTED.value}])

for task_info in all_tasks_info:
    task_info = api.task.stop(id=task_info['id'])
    supervisely.logger.info(f'{task_info}')

supervisely.logger.info(f'waiting for task(-s) to stop')

for task_info in all_tasks_info:
    task_info = api.task.wait(id=task_info['id'], target_status=api.task.Status.STOPPED.value)
    supervisely.logger.info(f'{task_info}')

supervisely.logger.info(f'{len(all_tasks_info)} task(-s) stopped')
