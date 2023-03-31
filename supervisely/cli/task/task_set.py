import supervisely as sly

import traceback
from rich.console import Console


def set_output_directory_run(task_id: int, team_id: int, dst_dir: str) -> bool:

    api = sly.Api.from_env()
    console = Console()

    if api.team.get_info_by_id(team_id) is None:
        console.print(
            f"\nError: Team with ID={team_id} is either not exist or not found in your acocunt\n",
            style="bold red",
        )
        return False
    if api.task.get_info_by_id(task_id) is None:
        console.print(
            f"\nError: Task with ID={task_id} is either not exist or not found in your acocunt\n",
            style="bold red",
        )
        return False
    if not api.file.dir_exists(team_id, dst_dir):
        console.print(f"\nError: directory '{dst_dir}' not exists in teamfiles\n", style="bold red")
        return False

    try:
        sly.output.set_directory(task_id, dst_dir)
        console.print("\nSetting task output directory succeed\n", style="bold green")
        return True

    except:
        console.print("\nSetting task output directory failed\n", style="bold red")
        traceback.print_exc()
        return False
