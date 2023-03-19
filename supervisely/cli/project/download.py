import os, shutil
import supervisely as sly

from dotenv import load_dotenv
from rich.console import Console

load_dotenv(os.path.expanduser("~/supervisely.env"))

api: sly.Api = sly.Api.from_env()

def download_project(project_id, save_dir):
    console = Console()
    console.print(f"\nDownloading data from {project_id} to directory: {save_dir}...\n", style="bold")

    if os.path.exists(save_dir):
        console.print(f"Deleting existing folder...")
        shutil.rmtree(save_dir)

    try:
        sly.download_project(api, project_id, save_dir)
        return True
    
    except Exception as e:
        console.print(f'Error: {e}')
        return False