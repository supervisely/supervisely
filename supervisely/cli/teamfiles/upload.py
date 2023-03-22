import os
import supervisely as sly
from functools import partial

from tqdm import tqdm

from dotenv import load_dotenv
from rich.console import Console


# debug only
# load_dotenv(os.path.expanduser("~/supervisely.env"))


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
    
    except Exception as e:
        print(f'Error: {e}')
        return False

def set_task_output_dir(team_id, task_id, dir):

    api = sly.Api.from_env()
       
    try:
        directory = api.file.list2(team_id, dir, recursive=False)
        
        if len(directory) == 0:
            print(f"Error: No files in teamfiles directory: {dir}")
            return False
        
        for elem in directory:
            elem_info = api.file.get_info_by_path(team_id, elem.path)

            if elem_info is None: # directory
                continue
            else: # file
                api.task.set_output_directory(task_id, elem_info.id, dir)
                return True

        print(f"Error: No files in teamfiles directory: {dir}")
        return False

    except Exception as e:
        print(f'Error: {e}')
        return False
    