import supervisely as sly
import traceback
from rich.console import Console
from tqdm import tqdm

def download(id, dest_dir):
    
    api = sly.Api.from_env()

    console = Console()
    console.print(f"\nDownloading data from project with ID={id} to directory: '{dest_dir}' ...\n", style="bold")

    project_info = api.project.get_info_by_id(id)

    if project_info is None:
        console.print('\nError: Project not exists\n', style="bold red")
        return False
            
    n_count = project_info.items_count
    try:
        with tqdm(total=n_count) as pbar:
            sly.download(api, id, dest_dir, progress_cb=pbar.update)

        console.print("\nProject is downloaded sucessfully!\n", style="bold green")
        return True
    except:
        console.print(f"\nProject is not downloaded\n", style='bold red')
        traceback.print_exc()
        return False