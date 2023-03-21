import os
import supervisely as sly


from dotenv import load_dotenv
from rich.console import Console


# debug only
# load_dotenv(os.path.expanduser("~/supervisely.env"))

def download_project(id, dir, type):
    
    api = sly.Api.from_env()
                
   
    console = Console()
    console.print(f"\nDownloading data from project with ID={id} to directory: '{dir}' ...\n", style="bold")
    
    # TODO Define log type (tqdm or stdout log info?)

    if type in ['images', 'img']:
        try:
            sly.download_project(api, id, dir, log_progress=True)
            return True
        except Exception as e:
            print(f'Error: {e}')
            return False
    
    elif type in ['videos', 'vid']:
        try:
            sly.download_video_project(api, id, dir, log_progress=True)
            return True
        except Exception as e:
            print(f'Error: {e}')
            return False

    elif type in ['volume project', 'volume', 'vol']:
        try:
            sly.download_volume_project(api, id, dir, log_progress=True)
            return True
        except Exception as e:
            print(f'Error: {e}')
            return False
        
    elif type in ['pointcloud project', 'pointcloud', 'ptcl']:
        try:
            sly.download_pointcloud_project(api, id, dir, log_progress=True)
            return True
        except Exception as e:
            print(f'Error: {e}')
            return False
        
    elif type in ['pointcloud episode project', 'pointcloud episode', 'ptclep']:
        try:
            sly.download_pointcloud_episode_project(api, id, dir, log_progress=True)
            return True
        except Exception as e:
            print(f'Error: {e}')
            return False
    else:
        print('Error: unknown type of project')
        return False