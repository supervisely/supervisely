import os
import supervisely as sly
from functools import partial

import json
from tqdm import tqdm

import time
import traceback
from rich.console import Console


def upload_to_teamfiles_run(team_id:int, local_dir:str, remote_dir:str) -> bool:
    
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
        
        console.print("\nLocal directory uploaded to teamfiles sucessfully!\n", style='bold green')
        return True
    
    except:
        console.print("\nUpload failed\n", style='bold red')
        traceback.print_exc()
        return False

def set_task_output_dir_run(team_id:int, task_id:int, dst_dir:str) -> bool:

    console = Console()
    api = sly.Api.from_env()
       
    try:
        files = api.file.list2(team_id, dst_dir, recursive=True)

        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": dst_dir}

            with open("/tmp/info.json", "w") as f:
                json.dump(data, f)

            src_path = os.path.join( os.getcwd(), "/tmp/info.json")
            dst_path = os.path.join( dst_dir, "/tmp/info.json")
            file_id = api.file.upload(team_id, src_path, dst_path).id

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, dst_dir)
        console.print("\nSetting task output directory succeed\n", style='bold green')
        return True

    except:
        console.print("\nSetting task output directory failed\n", style='bold red')
        traceback.print_exc()
        return False
    