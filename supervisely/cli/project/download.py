import supervisely as sly
import traceback
from rich.console import Console
from tqdm import tqdm

import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/supervisely.env"))

def download(id, dir):
    
    api = sly.Api.from_env()

    console = Console()
    console.print(f"\nDownloading data from project with ID={id} to directory: '{dir}' ...\n", style="bold")

    project_info = api.project.get_info_by_id(id)

    if project_info is None:
        print('Error: Project not exists')
        return False
            
    n_count = project_info.items_count
    try:
        with tqdm(total=n_count) as pbar:
            sly.download(api, id, dir, progress_cb=pbar.update)
        return True
    except:
        traceback.print_exc()
        return False