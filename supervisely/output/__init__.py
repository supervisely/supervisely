import json
import os

from supervisely._utils import is_production
from supervisely.api.api import Api
import supervisely.io.env as sly_env


def set_project(id: int):
    if is_production() is True:
        api = Api()
        task_id = sly_env.task_id()
        api.task.set_output_project(task_id, project_id=id)
    else:
        print(f"Output project: id={id}")


def set_directory(task_id: int, teamfiles_dir: str):
    if is_production() is True:
        api = Api()
        team_id = sly_env.team_id()

        files = api.file.list2(team_id, teamfiles_dir, recursive=True)
        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": teamfiles_dir}

            src_path = os.path.join(os.getcwd(), "info.json")
            with open(src_path, "w") as f:
                json.dump(data, f)

            dst_path = os.path.join(teamfiles_dir, "info.json")
            file_id = api.file.upload(team_id, src_path, dst_path).id

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, teamfiles_dir)
    else:
        print(f"Output directory for task with ID={task_id}: '{teamfiles_dir}'")
