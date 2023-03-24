import os
import supervisely as sly
from functools import partial
from dotenv import load_dotenv

import json
from tqdm import tqdm

import traceback
from rich.console import Console


def upload_to_teamfiles_run(team_id:int, local_dir:str, remote_dir:str) -> bool:

    if None in (os.environ.get('SERVER_ADDRESS'), os.environ.get('API_TOKEN')):
        load_dotenv(os.path.expanduser("~/supervisely.env"))

    console = Console()
    api = sly.Api.from_env()

    progress_bar = sly.app.widgets.Progress()

    if api.team.get_info_by_id(team_id) is None:
        console.print(f"\nError: Team with ID={team_id} not exists\n", style='bold red')
        return False         
    if not (os.path.exists(local_dir) and os.path.isdir(local_dir) ):
        console.print(f"\nError: local directory '{local_dir}' not exists\n", style='bold red')
        return False

    class ProgressBar(tqdm):
        def update_to(self, n: int) -> None:
            self.update(n - self.n)

    def upload_monitor_console(monitor, progress: sly.Progress, tqdm_pb: ProgressBar):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
            tqdm_pb.total = monitor.len
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
            tqdm_pb.update_to(monitor.bytes_read)

    def upload_monitor_instance(monitor, progress):
        readed_percent = round(monitor.bytes_read / monitor.len * 100)
        progress.update(readed_percent - progress.n)


    console.print(f"\nUploading local directory from '{local_dir}' to teamfiles directory: '{remote_dir}' ...\n", style="bold")

    try:
    
        if sly.is_development():
            progress = sly.Progress("Uploading training results directory to teamfiles...", 0, is_size=True)
            progress_size_cb = partial(upload_monitor_console, progress=progress, tqdm_pb=ProgressBar())
            api.file.upload_directory(team_id, local_dir, remote_dir, change_name_if_conflict=True, progress_size_cb=progress_size_cb)
        
        else:
            with progress_bar(message=f"Uploading training results directory to teamfiles...", total=100) as pbar:
                progress_size_cb = partial(upload_monitor_instance, progress=pbar)
                api.file.upload_directory(team_id, local_dir, remote_dir, change_name_if_conflict=True, progress_size_cb=progress_size_cb)
                pbar.message = 'Success'
                pbar.refresh()
                        
        return True
    
    except:
        console.print("\nUpload failed\n", style='bold red')
        traceback.print_exc()
        return False

def set_task_output_dir_run(task_id:int, team_id:int, dst_dir:str) -> bool:
    
    console = Console()
    api = sly.Api.from_env()

    if api.team.get_info_by_id(team_id) is None:
        console.print(f"\nError: Team with ID={team_id} not exists\n", style='bold red')
        return False     
    if api.task.get_info_by_id(task_id) is None:
        console.print(f"\nError: Task with ID={task_id} not exists\n", style='bold red')
        return False           
    if not api.file.dir_exists(team_id, dst_dir):
        console.print(f"\nError: directory '{dst_dir}' not exists in teamfiles\n", style='bold red')
        return False

       
    try:
        files = api.file.list2(team_id, dst_dir, recursive=True)

        if len(files) == 0:
            # some data to create dummy .json file to get file id
            data = {"team_id": team_id, "task_id": task_id, "directory": dst_dir}
            
            src_path = os.path.join( os.getcwd(), "info.json")
            with open(src_path, "w") as f:
                json.dump(data, f)
            
            dst_path = os.path.join( dst_dir, "info.json")
            file_id = api.file.upload( team_id, src_path, dst_path ).id

        else:
            file_id = files[0].id

        api.task.set_output_directory(task_id, file_id, dst_dir)
        console.print("\nSetting task output directory succeed\n", style='bold green')
        return True

    except:
        console.print("\nSetting task output directory failed\n", style='bold red')
        traceback.print_exc()
        return False
    
