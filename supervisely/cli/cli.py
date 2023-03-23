import sys
import click
from rich.console import Console


from supervisely.cli.download import download_run
from supervisely.cli.upload import upload_to_teamfiles_run, set_task_output_dir_run
from supervisely.cli.remove import remove_file_run, remove_dir_run
from supervisely.cli.get import get_project_name_run 



@click.group()
def cli():
    pass

@cli.command(
    help="This app allows you to release your aplication to Supervisely platform"
)
@click.option(
    "-p",
    "--path",
    required=False,
    help="[Optional] Path to the directory with application",
)
@click.option(
    "-a",
    "--sub-app",
    required=False,
    help="[Optional] Path to sub-app relative to application directory",
)
@click.option(
    "--release-version",
    required=False,
    help='[Optional] Release version in format "vX.X.X"',
)
@click.option("--release-description", required=False, help="[Optional] Release description (max length is 64 symbols)")
@click.option("-y", is_flag=True, help="[Optional] Add this flag for autoconfirm")
@click.option("-s", "--slug", required=False, help="[Optional] For internal use")
def release(path, sub_app, slug, y, release_version, release_description):
    from supervisely.cli.release import run
    try:
        success = run(
            app_directory=path,
            sub_app_directory=sub_app,
            slug=slug,
            autoconfirm=y,
            release_version=release_version,
            release_description=release_description,
        )
        if success:
            print("App released sucessfully!")
            sys.exit(0)
        else:
            print("App not released")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Aborting...")
        print("App not released")
        sys.exit(1)


@cli.command(
    help="Get project name"
)
@click.option(
    "-id",
    "--id",
    required=True,
    type=int,
    help="Supervisely project ID",
)
def get_project_name(id:int) -> None:    
    try:
        success = get_project_name_run(id)
        if success:
            sys.exit(0)
        else:
            print("Getting project name failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Getting project name directory aborted")
        sys.exit(1)

@cli.command(
    help="Download project data from supervisely to local directory"
)
@click.option(
    "-id",
    "--project-id",
    required=True,
    type=int,
    help="Supervisely project ID",
)
@click.option(
    "-d",
    "--dst",
    required=True,
    type=str,
    help="Download destination directory",
)
def download(project_id:id, dst:str) -> None:
    try:
        success = download_run(project_id, dst)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload aborted\n")
        sys.exit(1)


@cli.command(
    help="Remove file from supervisely teamfiles"
)
@click.option(
    "-id",
    "--team-id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "-p",
    "--path",
    required=True,
    type=str,
    help="File path to remove",
)
def remove_file(team_id:int, path:str) -> None:
    try:
        success = remove_file_run(team_id, path)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nRemoving file aborted\n")
        sys.exit(1)

@cli.command(
    help="Remove directory from supervisely teamfiles"
)
@click.option(
    "-id",
    "--team-id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "-p",
    "--path",
    required=True,
    type=str,
    help="Path to remove directory",
)
def remove_dir(team_id:int, path:str) -> None:
    try:
        success = remove_dir_run(team_id, path)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Removing directory aborted")
        sys.exit(1)


@cli.command(
    help="Upload local source files with destination to supervisely teamfiles"
)
@click.option(
    "-id",
    "--team-id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "-s",
    "--src",
    required=True,
    type=str,
    help="Path to local source directory from which files are uploaded",
)
@click.option(
    "-d",
    "--dst",
    required=True,
    type=str,
    help="Path to teamfiles remote destination directory to which files are uploaded",
)
def upload(team_id:int, src:str, dst:str) -> None:
    console = Console()
    try:
        success = upload_to_teamfiles_run(team_id, src, dst)
        if success:
            console.print("\nLocal directory uploaded to teamfiles sucessfully!\n", style='bold green')
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Upload aborted")
        sys.exit(1)
        
@cli.command(
    help="Set link to teamfiles directory at workspace tasks interface"
)
@click.option(
    "-id",
    "--team-id",
    required=True,
    type=int,
    help="Supervisely team ID",
)
@click.option(
    "--task-id",
    required=True,
    type=int,
    help="Supervisely task ID",
)
@click.option(
    "-d",
    "--dir",
    required=True,
    type=str,
    help="Path to teamfiles directory",
)
def set_task_output_dir(team_id:int, task_id:int, dir:str) -> None:
    try:
        success = set_task_output_dir_run(team_id, task_id, dir)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Setting task output directory aborted")
        sys.exit(1)
