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
        console.print(f"\nError: Team with ID={team_id} not exists\n", style="bold red")
        return False
    if not os.path.isdir(local_dir):
        console.print(f"\nError: local directory '{local_dir}' not exists\n", style="bold red")
        return False

    console.print(
        f"\nUploading local directory from '{local_dir}' to Team files directory: '{remote_dir}' ...\n",
        style="bold",
    )

    try:
        if sly.is_development():

            def upload_monitor_console(tqdm_pb, monitor):
                if tqdm_pb.total == 0:
                    tqdm_pb.total = monitor.len
                tqdm_pb.update(monitor.bytes_read - tqdm_pb.n)
                if monitor.bytes_read == monitor.len:
                    tqdm_pb.refresh()  # refresh progress bar to show completion
                    tqdm_pb.close()  # close progress bar

            print("Please wait ...")
            pbar = tqdm(total=0, unit="B", unit_scale=True)
            progress_size_cb = partial(upload_monitor_console, pbar)

            time.sleep(1)

        else:

            def upload_monitor_instance(monitor, progress: sly.Progress):
                if progress.total == 0:
                    progress.set(monitor.bytes_read, monitor.len, report=False)
                else:
                    progress.set_current_value(monitor.bytes_read, report=False)
                    if progress.need_report():
                        progress.report_progress()

            pbar = sly.Progress("Uploading local directory to Team files...", 0, is_size=True)
            progress_size_cb = partial(upload_monitor_instance, progress=pbar)

        res_remote_dir = api.file.upload_directory(
            team_id,
            local_dir,
            remote_dir,
            change_name_if_conflict=True,
            progress_size_cb=progress_size_cb,
        )

        if res_remote_dir != remote_dir:
            console.print(
                f"\nWarning: Team files directory '{remote_dir}' already exists. Name changed to '{res_remote_dir}'.",
                style="bold red",
            )

        console.print("\nLocal directory uploaded to teamfiles sucessfully!\n", style="bold green")
        return True

    except:
        console.print("\nUpload failed\n", style="bold red")
        traceback.print_exc()
        return False
