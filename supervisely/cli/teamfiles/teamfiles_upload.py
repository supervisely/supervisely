import os
import supervisely as sly
from functools import partial

from tqdm import tqdm
import time

import traceback
from rich.console import Console


def upload_directory_run(team_id: int, local_dir: str, remote_dir: str) -> bool:
    """
    Note: to extract actual res_remote_dir (==$TEAMFILES_DIR) in case of name conflict (change_name_if_conflict=True) use following bash command:

    ```bash
        output=$(supervisely teamfiles upload -id $TEAM_ID --src "/src/path/" --dst "/dst/path/" | tee /dev/tty)

        string=$(echo $output | grep -o "Local directory was sucessfully uploaded to Team files directory: '[^']*'")
        TEAMFILES_DIR=$(echo $string | grep -o "'[^']*'" | sed "s/'//g")

        if [ "$TEAMFILES_DIR" != "/dst/path/"  ]
        then
            echo "local and remote directories not matching!" #do your code here
            echo "Actual Team files directory: $TEAMFILES_DIR"
        fi
    """

    api = sly.Api.from_env()
    console = Console()

    if api.team.get_info_by_id(team_id) is None:
        console.print(
            f"\nError: Team with ID={team_id} is either not exist or not found in your acocunt\n",
            style="bold red",
        )
        return False
    if not os.path.isdir(local_dir):
        console.print(f"\nError: local directory '{local_dir}' not exists\n", style="bold red")
        return False

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

        res_remote_dir = api.file.upload_directory(
            team_id,
            local_dir,
            remote_dir,
            change_name_if_conflict=True,
            progress_size_cb=progress_size_cb,
        )

        if res_remote_dir != remote_dir:
            console.print(
                f"\nWarning: '{remote_dir}' already exists. Creating a new directory in Team files: '{res_remote_dir}'",
                style="bold yellow",
            )
        else:
            res_remote_dir = remote_dir

        console.print(
            f"\nLocal directory was sucessfully uploaded to Team files with following path: '{res_remote_dir}'.\n",
            style="bold green",
        )
        return True

    except:
        console.print("\nUpload failed\n", style="bold red")
        traceback.print_exc()
        return False