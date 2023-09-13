import traceback

from rich.console import Console
from tqdm import tqdm
import supervisely as sly
from dotenv import load_dotenv
import os


def download_run(id: int, dest_dir: str) -> bool:
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
