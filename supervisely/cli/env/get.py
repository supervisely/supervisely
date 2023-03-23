import supervisely as sly
import click

import traceback

def get_project_name(project_id:int) -> bool:

    api = sly.Api.from_env()

    try:
        project_info = api.project.get_info_by_id(project_id)
        click.echo(project_info.name)
        return True
    
    except:
        traceback.print_exc()
        return False
    

