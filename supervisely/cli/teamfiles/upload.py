import supervisely as sly
from functools import partial

from tqdm import tqdm

import traceback
from rich.console import Console

import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/supervisely.env"))


def upload_to_teamfiles(team_id, local_dir, remote_dir):
    
    class ProgressBar(tqdm):
        def update_to(self, n: int) -> None:
            self.update(n - self.n)

    def upload_monitor(monitor, progress: sly.Progress, tqdm_pb: ProgressBar):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
            tqdm_pb.total = monitor.len
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
            tqdm_pb.update_to(monitor.bytes_read)
                
    progress = sly.Progress("Upload artefacts directory to teamfiles...", 0, is_size=True)
    progress_size_cb = partial(upload_monitor, progress=progress, tqdm_pb=ProgressBar())

    api = sly.Api.from_env()

    console = Console()
    console.print(f"\nUploading local directory from '{local_dir}' to teamfiles directory: '{remote_dir}' ...\n", style="bold")

    try:
        api.file.upload_directory(
            team_id, local_dir, remote_dir,
            change_name_if_conflict=True,
            progress_size_cb=progress_size_cb
        )
        return True
    
    except:
        traceback.print_exc()
        return False

def set_task_output_dir(team_id, task_id, dir):

    api = sly.Api.from_env()
       
    try:
        files = api.file.list2(team_id, dir, recursive=True)

        breakpoint()
        
        if len(files) == 0:
            print(f"Error: No files in teamfiles directory: {dir}")
            return False

        api.task.set_output_directory(task_id, files[0].id, dir)

        # for file in files:
        #     info = api.file.get_info_by_path(team_id, file.path)

        #     api.file.

        #     if info is None: # directory
        #         continue
        #     else: # file
        #         api.task.set_output_directory(task_id, info.id, dir)
        #         return True

        # print(f"Error: No files in teamfiles directory: {dir}")
        # return False

    except:
        traceback.print_exc()
        return False
    