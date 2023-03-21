import os
import supervisely as sly

from dotenv import load_dotenv

# debug only
# load_dotenv(os.path.expanduser("~/supervisely.env"))


def remove_(team_id, path):

    api = sly.Api.from_env()

    try:
        file_info = api.file.get_info_by_path(team_id, path)

        if file_info.ext is None:
            print('Path extension not exists. Maybe you entered directory?')
            return False
        
        api.file.remove(team_id, path)
        print(f"Path '{path}' removed")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def remove_dir(team_id, path):

    api = sly.Api.from_env()
    
    try:
        if api.file.dir_exists(team_id, path):
            api.file.remove(team_id, path)
            print(f"Directory '{path}' removed")
            return True
        
        print('Directory not exists.')
        return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False

