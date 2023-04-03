import json
import os

from supervisely._utils import is_production
from supervisely.api.api import Api
import supervisely.io.env as sly_env
from supervisely import rand_str


def set_project(id: int):
    if is_production() is True:
        api = Api()
        task_id = sly_env.task_id()
        api.task.set_output_project(task_id, project_id=id)
    else:
        print(f"Output project: id={id}")


def set_directory(teamfiles_dir: str):
    if is_production():

        api = Api()
        task_id: sly_env.task_id()
        team_id = api.task.get_info_by_id(task_id)["teamId"]

        files = api.file.list2(team_id, teamfiles_dir, recursive=True)

        # if directory is empty or not exists
        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": teamfiles_dir}
            filename = f"info.json"

            src_path = os.path.join("/tmp/", filename)
            with open(src_path, "w") as f:
                json.dump(data, f)

            dst_path = os.path.join(teamfiles_dir, filename)
            file_id = api.file.upload(team_id, src_path, dst_path).id

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, teamfiles_dir)

        # remove dummy file
        if len(files) == 0:
            api.file.remove_file(team_id, dst_path)

    else:
        print(f"Output directory: '{teamfiles_dir}'")
