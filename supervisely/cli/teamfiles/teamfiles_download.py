import os
import supervisely as sly

from tqdm import tqdm

import traceback
from rich.console import Console


def download_directory_run(
    team_id: int,
    remote_dir: str,
    local_dir: str,
    filter: str = None,
    ignore_if_not_exists: bool = False,
) -> bool:
    api = sly.Api.from_env()
    console = Console()

    if api.team.get_info_by_id(team_id) is None:
        console.print(
            f"\nError: Team with ID={team_id} is either not exist or not found in your acocunt\n",
            style="bold red",
        )
        return False

    # force directories to end with slash '/'
    if not local_dir.endswith(os.path.sep):
        local_dir = os.path.join(local_dir, "")
    if not remote_dir.endswith("/"):
        remote_dir += "/"

    if not ignore_if_not_exists:
        files = api.file.list2(team_id, remote_dir, recursive=True)
        if len(files) == 0:
            console.print(
                f"\nError:  Team files folder '{remote_dir}' not exists\n", style="bold red"
            )
            return False

    console.print(
        f"\nDownloading directory '{remote_dir}' from Team files ...\n",
        style="bold",
    )

    total_size = api.file.get_directory_size(team_id, remote_dir)
    p = tqdm(desc="Downloading...", total=total_size, unit="B", unit_scale=True)
    try:
        api.file.download_directory(team_id, remote_dir, local_dir, progress_cb=p)

        console.print(
            f"\nTeam files directory was sucessfully downloaded to the local path: '{local_dir}'.\n",
            style="bold green",
        )
        return True

    except:
        console.print("\nDownload failed\n", style="bold red")
        traceback.print_exc()
        return False
