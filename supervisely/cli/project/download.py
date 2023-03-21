import os, shutil
import supervisely as sly

from dotenv import load_dotenv
from rich.console import Console

load_dotenv(os.path.expanduser("~/supervisely.env"))

def download_project(project_id, dir):

    api = sly.Api.from_env()
    
    console = Console()
    console.print(f"\nDownloading data from project with ID={project_id} to directory: '{dir}' ...\n", style="bold")

    try:
        # TODO progress
        sly.download_project(api, project_id, dir)
        return True
    
    except Exception as e:
        print(f'Error: {e}')
        return False