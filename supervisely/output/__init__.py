import json
import os

from supervisely._utils import is_production
from supervisely.api.api import Api
import supervisely.io.env as sly_env
from supervisely import rand_str
from supervisely.io.fs import silent_remove


def set_project(id: int):
    if is_production() is True:
        api = Api()
        task_id = sly_env.task_id()
        api.task.set_output_project(task_id, project_id=id)
    else:
        print(f"Output project: id={id}")


def set_directory(teamfiles_dir: str):
    """
    Sets a link to a teamfiles directory in workspace tasks interface
    """
    if is_production():

        api = Api()
        task_id = sly_env.task_id()

        if api.task.get_info_by_id(task_id) is None:
            raise KeyError(
                f"Task with ID={task_id} is either not exist or not found in your account"
            )

        team_id = api.task.get_info_by_id(task_id)["teamId"]

        if api.team.get_info_by_id(team_id) is None:
            raise KeyError(
                f"Team with ID={team_id} is either not exist or not found in your account"
            )

        files = api.file.list2(team_id, teamfiles_dir, recursive=True)

        # if directory is empty or not exists
        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": teamfiles_dir}
            filename = f"{rand_str(10)}.json"

            src_path = os.path.join("/tmp/", filename)
            with open(src_path, "w") as f:
                json.dump(data, f)

            dst_path = os.path.join(teamfiles_dir, filename)
            file_id = api.file.upload(team_id, src_path, dst_path).id

            silent_remove(src_path)

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, teamfiles_dir)

    else:
        print(f"Output directory: '{teamfiles_dir}'")
