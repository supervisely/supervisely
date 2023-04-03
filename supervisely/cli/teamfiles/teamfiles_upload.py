import os
import supervisely as sly
from functools import partial

from tqdm import tqdm
import time

import traceback
from rich.console import Console


def upload_directory_run(team_id: int, local_dir: str, remote_dir: str) -> bool:

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
    if not remote_dir.endswith(os.path.sep):
        remote_dir = os.path.join(remote_dir, "")

    if not os.path.isdir(local_dir):
        console.print(f"\nError: local directory '{local_dir}' not exists\n", style="bold red")
        return False

    files = api.file.list2(team_id, remote_dir, recursive=True)
    if len(files) > 0:
        if files[0].path.startswith(remote_dir):
            console.print(
                f"\nError: Team files folder '{remote_dir}' already exists. Please enter unique path for your folder.\n",
                style="bold red",
            )
            return False
    else:
        pass  # new folder

    console.print(
        f"\nUploading local directory '{local_dir}' to Team files ...\n",
        style="bold",
    )
    try:
        if sly.is_development():

            def upload_monitor_console(monitor, progress: tqdm):
                if progress.total == 0:
                    progress.total = monitor.len
                progress.update(monitor.bytes_read - progress.n)
                if monitor.bytes_read == monitor.len:
                    progress.refresh()  # refresh progress bar to show completion
                    progress.close()  # close progress bar

            # api.file.upload_directory may be slow depending on the number of folders
            print("Please wait ...")

            progress = tqdm(total=0, unit="B", unit_scale=True)
            progress_size_cb = partial(upload_monitor_console, progress=progress)

            time.sleep(1)  # for better UX

        else:

            def upload_monitor_instance(monitor, progress: sly.Progress):
                if progress.total == 0:
                    progress.set(monitor.bytes_read, monitor.len, report=False)
                else:
                    progress.set_current_value(monitor.bytes_read, report=False)
                    if progress.need_report():
                        progress.report_progress()

            progress = sly.Progress("Uploading to Team files...", 0, is_size=True)
            progress_size_cb = partial(upload_monitor_instance, progress=progress)

        # no need in change_name_if_conflict due to previous exception handling
        api.file.upload_directory(
            team_id,
            local_dir,
            remote_dir,
            progress_size_cb=progress_size_cb,
        )

        console.print(
            f"\nLocal directory was sucessfully uploaded to Team files with following path: '{remote_dir}'.\n",
            style="bold green",
        )
        return True

    except:
        console.print("\nUpload failed\n", style="bold red")
        traceback.print_exc()
        return False
