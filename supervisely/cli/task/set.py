import os
import supervisely as sly

import json

import traceback
from rich.console import Console


def set_task_output_dir_run(task_id: int, team_id: int, dst_dir: str) -> bool:

      
    api = sly.Api.from_env()
    console = Console()

    if api.team.get_info_by_id(team_id) is None:
        console.print(f"\nError: Team with ID={team_id} not exists\n", style="bold red")
        return False
    if api.task.get_info_by_id(task_id) is None:
        console.print(f"\nError: Task with ID={task_id} not exists\n", style="bold red")
        return False
    if not api.file.dir_exists(team_id, dst_dir):
        console.print(f"\nError: directory '{dst_dir}' not exists in teamfiles\n", style="bold red")
        return False

    try:
        files = api.file.list2(team_id, dst_dir, recursive=True)

        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": dst_dir}

            src_path = os.path.join(os.getcwd(), "info.json")
            with open(src_path, "w") as f:
                json.dump(data, f)

            dst_path = os.path.join(dst_dir, "info.json")
            file_id = api.file.upload(team_id, src_path, dst_path).id

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, dst_dir)
        console.print("\nSetting task output directory succeed\n", style="bold green")
        return True

    except:
        console.print("\nSetting task output directory failed\n", style="bold red")
        traceback.print_exc()
        return False
