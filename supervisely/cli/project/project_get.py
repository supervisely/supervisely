import traceback

from rich.console import Console

import supervisely as sly
from dotenv import load_dotenv
import os


def get_project_name_run(project_id: int) -> bool:
    console = Console()

    # get server address
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    server_address = os.getenv("SERVER_ADDRESS", None)
    if server_address is None:
        console.print(
            '[red][Error][/] Cannot find [green]SERVER_ADDRESS[/]. Add it to your "~/supervisely.env" file or to environment variables'
        )
        return False

    # get api token
    api_token = os.getenv("API_TOKEN", None)
    if api_token is None:
        console.print(
            '[red][Error][/] Cannot find [green]API_TOKEN[/]. Add it to your "~/supervisely.env" file or to environment variables'
        )
        return False

    api = sly.Api.from_env()

    try:
        project_info = api.project.get_info_by_id(project_id)
        if project_info is None:
            print(
                f"Project with ID={project_id} is either archived, doesn't exist or you don't have enough permissions to access it"
            )
            return False

        print(project_info.name)
        return True

    except:
        traceback.print_exc()
        return False
