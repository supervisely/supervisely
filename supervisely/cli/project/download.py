import os
import supervisely as sly

import time

from dotenv import load_dotenv
from rich.console import Console

from tqdm import tqdm

# debug only
# load_dotenv(os.path.expanduser("~/supervisely.env"))

def download_project(id, dir):
    
    api = sly.Api.from_env()

    console = Console()
    console.print(f"\nDownloading data from project with ID={id} to directory: '{dir}' ...\n", style="bold")

    # get count of elements (works for all types)
    n_count = api.project.get_images_count(id)

    type = api.project.get_info_by_id(id).type
    if type == 'images':
        try:
            with tqdm(total=n_count) as pbar:
                sly.download_project(api, id, dir, progress_cb=pbar.update)
            return True
        
        except Exception as e:
            print(f'Error: {e}')
            return False
    
    elif type == 'videos':
        try:
            with tqdm(total=n_count) as pbar:
                sly.download_video_project(api, id, dir, progress_cb=pbar.update)
            return True
        
        except Exception as e:
            print(f'Error: {e}')
            return False

    elif type == 'volumes':
        try:
            sly.download_volume_project(api, id, dir, log_progress=True)
            return True

        except Exception as e:
            print(f'Error: {e}')
            return False
        
    elif type == 'point_clouds':
        try:
            with tqdm(total=n_count) as pbar:
                sly.download_pointcloud_project(api, id, dir, progress_cb=pbar.update)
            return True
        
        except Exception as e:
            print(f'Error: {e}')
            return False
        
    elif type == 'point_cloud_episodes':
        try:
            with tqdm(total=n_count) as pbar:
                sly.download_pointcloud_episode_project(api, id, dir, progress_cb=pbar.update)
            return True
        
        except Exception as e:
            print(f'Error: {e}')
            return False
    else:
        print(f'Error: unknown type of project ({type})')
        return False