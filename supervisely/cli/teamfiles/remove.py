import supervisely as sly
import traceback

import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/supervisely.env"))


def remove_file(team_id, path):

    api = sly.Api.from_env()

    try:
        api.file.remove_file(team_id, path)
        return True
    
    except:
        traceback.print_exc()
        return False
    
def remove_dir(team_id, path):

    api = sly.Api.from_env()
    
    try:
        api.file.remove_dir(team_id, path)
        return True
    
    except:
        traceback.print_exc()
        return False
