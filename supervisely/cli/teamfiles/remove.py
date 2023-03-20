import os
import supervisely as sly

from dotenv import load_dotenv
from rich.console import Console

load_dotenv(os.path.expanduser("~/supervisely.env"))

api: sly.Api = sly.Api.from_env()

def remove(team_id, path):
    
    console = Console()

    try:
        api.file.remove(team_id, path)

        console.print(f"Directory '{path}' removed")
        return True
    
    except Exception as e:
        console.print(f"Error: {e}")
        return False

