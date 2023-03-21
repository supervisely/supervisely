import os
import supervisely as sly

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/supervisely.env"))


def remove_(team_id, path):

    api = sly.Api.from_env()

    try:
        api.file.remove(team_id, path)

        print(f"Directory '{path}' removed")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def remove_dir(team_id, path):

    api = sly.Api.from_env()
    
    try:
        api.file.remove(team_id, path)

        print(f"Directory '{path}' removed")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

