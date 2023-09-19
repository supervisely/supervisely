import traceback

from rich.console import Console
import tqdm
import supervisely as sly
from dotenv import load_dotenv
import os


def download_run(id: int, dest_dir: str) -> bool:
    console = Console()

    api = sly._handle_creds_error_to_console(sly.Api.from_env, console.print)
    if not api:
        return False

    console.print(
        f"\nDownloading data from project with ID={id} to directory: '{dest_dir}' ...\n",
        style="bold",
    )

    project_info = api.project.get_info_by_id(id)
    if project_info is None:
        console.print(f"\nError: Project '{project_info}' not exists\n", style="bold red")
        return False

    n_count = project_info.items_count
    try:
        if sly.is_development():
            with tqdm.tqdm(total=n_count) as pbar:
                sly.download(api, id, dest_dir, progress_cb=pbar)
        else:
            sly.download(api, id, dest_dir, log_progress=True)

        console.print(
            f"\nProject '{project_info.name}' is downloaded sucessfully!\n", style="bold green"
        )
        return True
    except:
        console.print(f"\nProject '{project_info.name}' is not downloaded\n", style="bold red")
        traceback.print_exc()
        return False
