import traceback

from rich.console import Console
from tqdm import tqdm
import supervisely as sly
from supervisely.io.fs import dir_exists

from dotenv import load_dotenv
import os


def upload_run(src_dir: str, workspace_id: int, project_name: str = None) -> bool:
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

    project_fs = sly.read_project(src_dir)
    if project_name is None:
        project_name = project_fs.name

    console.print(
        f"\nUploading data from the source directory: '{src_dir}' ...\n",
        style="bold",
    )

    if not dir_exists(src_dir):
        console.print(f"\nError: Directory '{src_dir}' not exists\n", style="bold red")
        return False

    if api.workspace.get_info_by_id(workspace_id) is None:
        console.print(f"\nError: Workspace with id={workspace_id} not exists\n", style="bold red")

    try:
        if sly.is_development():
            with tqdm(total=project_fs.total_items) as pbar:
                sly.upload(src_dir, api, workspace_id, project_name, progress_cb=pbar.update)
                pbar.refresh()
        else:
            sly.upload(src_dir, api, workspace_id, project_name, log_progress=True)

        console.print(f"\nProject '{project_name}' is uploaded sucessfully!\n", style="bold green")
        return True
    except:
        console.print(f"\nProject is not uploaded\n", style="bold red")
        traceback.print_exc()
        return False
