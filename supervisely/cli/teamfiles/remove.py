import supervisely as sly
import traceback

import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/supervisely.env"))


def remove_file(team_id, path):

    api = sly.Api.from_env()

    try:
        file_info = api.file.get_info_by_path(team_id, path)

        if file_info is None:
            print(f'Not a file. Maybe you entered directory? (Path: {path})')
            return False
        
        api.file.remove(team_id, path)
        return True
    
    except:
        traceback.print_exc()
        return False
    
def remove_dir(team_id, path):

    api = sly.Api.from_env()
    
    try:
        if api.file.dir_exists(team_id, path):
            api.file.remove(team_id, path)
            return True
        
        print(f"Directory '{path}' not exists.")
        return False
    
    except Exception as e:
        traceback.print_exc()
        return False
