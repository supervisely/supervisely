import supervisely as sly
import traceback

from rich.console import Console


def remove_file_run(team_id:int, path:str) -> bool:

    console = Console()
    api = sly.Api.from_env()

    try:
        api.file.remove_file(team_id, path)
        console.print(f"\nFile '{path}' successfully removed\n", style='bold green')
        return True
    
    except:
        console.print(f"\nRemoving file failed\n", style='bold red')
        traceback.print_exc()
        return False
    
def remove_dir_run(team_id:int, path:str) -> bool:

    console = Console()
    api = sly.Api.from_env()
    
    try:
        api.file.remove_dir(team_id, path)
        console.print(f"\nDirectory '{path}' successfully removed\n", style='bold green')
        return True
    
    except:
        console.print(f"\nRemoving directory failed\n", style='bold red')
        traceback.print_exc()
        return False
