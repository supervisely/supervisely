import supervisely as sly
import click

import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/supervisely.env"))


def get_project_name(project_id, replace_space=False):

    api = sly.Api.from_env()
    try:
        project_info = api.project.get_info_by_id(project_id)
        click.echo(f"{project_info.name}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def get_synced_dir():
    try:
        synced_dir = sly.app.get_synced_data_dir()
        click.echo(synced_dir)
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

