import supervisely as sly
import supervisely.io.env as sly_env
from supervisely._utils import is_production

import traceback
from rich.console import Console


def set_output_directory_run(dst_dir: str) -> bool:

    console = Console()

    if is_production():

        api = sly.Api.from_env()
        task_id: sly_env.task_id()

        if api.task.get_info_by_id(task_id) is None:
            console.print(
                f"\nError: Task with ID={task_id} is either not exist or not found in your acocunt\n",
                style="bold red",
            )
            return False

        team_id = api.task.get_info_by_id(task_id)["teamId"]

        if api.team.get_info_by_id(team_id) is None:
            console.print(
                f"\nError: Team with ID={team_id} is either not exist or not found in your acocunt\n",
                style="bold red",
            )
            return False

        if not api.file.dir_exists(team_id, dst_dir):
            console.print(
                f"\nError: directory '{dst_dir}' not exists in teamfiles\n", style="bold red"
            )
            return False

    try:
        sly.output.set_directory(dst_dir)
        console.print("\nSetting task output directory succeed\n", style="bold green")
        return True

    except:
        console.print("\nSetting task output directory failed\n", style="bold red")
        traceback.print_exc()
        return False
