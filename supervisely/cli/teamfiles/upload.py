import os
import supervisely as sly
from functools import partial

from dotenv import load_dotenv
from rich.console import Console

load_dotenv(os.path.expanduser("~/supervisely.env"))


def upload_to_teamfiles(team_id, local_dir, remote_dir):

    api = sly.Api.from_env()

    console = Console()
    console.print(f"\nUploading local directory from '{local_dir}' to teamfiles directory: '{remote_dir}' ...\n", style="bold")

    try:
        # files = api.file.listdir(team_id, local_dir, recursive=False)
        # progress = sly.Progress(message='Uploading...', total_cnt=len(files))
        # progress_cb = partial(upload_monitor, api, task_id, progress)

        api.file.upload_directory(
            team_id, local_dir, remote_dir,
            change_name_if_conflict=True,
            # progress_size_cb=progress_cb
        )
        return True
    
    except Exception as e:
        print(f'Error: {e}')
        return False

def set_task_output_dir(team_id, task_id, dir):

    api = sly.Api.from_env()
       
    try:
        files = api.file.list2(team_id, dir, recursive=False)
        
        if len(files) == 0:
            print(f"Error: No files in teamfiles directory: {dir}")
            return False
        
        file_info = api.file.get_info_by_path(
            team_id,
            files[0].path
        )

        api.task.set_output_directory(task_id, file_info.id, dir)
        return True
    
    except Exception as e:
        print(f'Error: {e}')
        return False
    