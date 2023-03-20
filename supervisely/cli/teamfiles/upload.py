import os
import supervisely as sly

from dotenv import load_dotenv
from rich.console import Console

load_dotenv(os.path.expanduser("~/supervisely.env"))

api: sly.Api = sly.Api.from_env()

def upload_to_teamfiles(team_id, from_local_dir, to_teamfiles_dir):
    
    console = Console()
    console.print(f"\nUploading local directory from '{from_local_dir}' to teamfiles directory: '{to_teamfiles_dir}' ...\n", style="bold")

    try:
        api.file.upload_directory(
            team_id, from_local_dir, to_teamfiles_dir,
            change_name_if_conflict=True
        )
        return True
    
    except Exception as e:
        console.print(f'Error: {e}')
        return False

def set_task_output_dir(team_id, task_id, teamfiles_dir):
    
    console = Console()
    
    try:
        files = api.file.list(team_id, teamfiles_dir)
        
        if len(files) == 0:
            console.print(f"Error: No files in teamfiles directory: {teamfiles_dir}")
            return False
        
        file_info = api.file.get_info_by_path(
            team_id,
            files[0]['path']
        )

        api.task.set_output_directory(task_id, file_info.id, teamfiles_dir)
        return True
    
    except Exception as e:
        console.print(f'Error: {e}')
        return False
    