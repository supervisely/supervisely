import supervisely
from supervisely.api.task_api import TaskStatuses

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
                                                                        'value': TaskStatuses.STARTED}])

for task_info in all_tasks_info:
    task_info = api.task.stop(id=task_info['id'])
    supervisely.logger.info(f'{task_info=}')

supervisely.logger.info(f'{len(all_tasks_info)} task(-s) stopped')
