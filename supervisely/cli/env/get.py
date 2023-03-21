import os, re
import supervisely as sly
import click
from dotenv import load_dotenv

# debug only
# load_dotenv(os.path.expanduser("~/supervisely.env"))


def get_project_name(project_id, replace_space=False):

    def strip_with_replacement(text, replacement):
        return re.sub(r'\s+', replacement, text)
    
    api = sly.Api.from_env()

    try:
        project_info = api.project.get_info_by_id(project_id)

        if replace_space:
            project_name = strip_with_replacement(project_info.name, '_')
        else:
            project_name = project_info.name

        click.echo(f"{project_name}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def get_synced_dir():

    try:
        synced_dir = sly.app.get_synced_data_dir()

        click.echo(f"{synced_dir}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

