import traceback

from rich.console import Console
from tqdm import tqdm
import supervisely as sly
from dotenv import load_dotenv
import os


def download_run(id: int, dest_dir: str) -> bool:
    console = Console()

    load_dotenv(os.path.expanduser("~/supervisely.env"))
    try:
        api = sly.Api.from_env()
    except ValueError as e:
        console.print(
            f'{e}\nAdd it to your "~/supervisely.env" file or to environment variables',
            style="bold red",
        )
        return False

    console.print(
        f"\nDownloading data from project with ID={id} to directory: '{dest_dir}' ...\n",
        style="bold",
    )

    project_info = api.project.get_info_by_id(id)

    if project_info is None:
        console.print("\nError: Project not exists\n", style="bold red")
        return False

    n_count = project_info.items_count
    try:
        if sly.is_development():
            with tqdm(total=n_count) as pbar:
                sly.download(api, id, dest_dir, progress_cb=pbar.update)
                pbar.refresh()
        else:
            sly.download(api, id, dest_dir, log_progress=True)

        console.print("\nProject is downloaded sucessfully!\n", style="bold green")
        return True
    except:
        console.print(f"\nProject is not downloaded\n", style="bold red")
        traceback.print_exc()
        return False
