import supervisely as sly
import click

import traceback

def get_project_name_run(project_id:int) -> bool:
    # TODO

    api = sly.Api.from_env()

    try:
        project_info = api.project.get_info_by_id(project_id)
        if project_info is None:
            click.echo(f"Project with id {project_id} is either archived, doesn't exist or you don't have enough permissions to access it")
            return False
        print(project_info.name)
        return True
    
    except:
        traceback.print_exc()
        return False
    

