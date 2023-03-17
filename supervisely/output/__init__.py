from supervisely._utils import is_production
import supervisely.io.env as sly_env
from supervisely.api.api import Api


def set_project(id: int):
    if is_production() is True:
        api = Api()
        task_id = sly_env.task_id()
        api.task.set_output_project(task_id, project_id=id)
    else:
        print(f"Output project: id={id}")
