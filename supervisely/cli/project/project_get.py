import traceback

from rich.console import Console

import supervisely as sly
from dotenv import load_dotenv
import os


def get_project_name_run(project_id: int) -> bool:
    console = Console()

    load_dotenv(os.path.expanduser("~/supervisely.env"))
    try:
        api = sly.Api.from_env()
    except KeyError as e:
        console.print(
            f"Error: {e}\n\nAdd it to your '~/supervisely.env' file or to environment variables",
            style="bold red",
        )
        return False

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
