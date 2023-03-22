import click, sys


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
    help="Download project data from supervisely to local directory"
)
@click.option(
    "-id",
    "--project-id",
    required=True,
    help="Supervisely project ID",
)
@click.option(
    "-d",
    "--dst",
    required=True,
    help="Download destination directory",
)
def download(project_id, dst):
    from supervisely.cli.project import download
    try:
        success = download(project_id, dst)
        if success:
            print("Project is downloaded sucessfully!")
            sys.exit(0)
        else:
            print(f"Project is not downloaded")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Download aborted")
        sys.exit(1)


@cli.command(
    help="Remove file from supervisely teamfiles"
)
@click.option(
    "-id",
    "--team-id",
    required=True,
    help="Supervisely team ID",
)
@click.option(
    "-p",
    "--path",
    required=True,
    help="File path to remove",
)
def remove_file(team_id, path):
    from supervisely.cli.teamfiles import remove_file
    try:
        success = remove_file(team_id, path)
        if success:
            print(f"File '{path}' successfully removed")
            sys.exit(0)
        else:
            print(f"Removing file failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Removing file aborted")
        sys.exit(1)

@cli.command(
    help="Remove directory from supervisely teamfiles"
)
@click.option(
    "-id",
    "--team-id",
    required=True,
    help="Supervisely team ID",
)
@click.option(
    "-p",
    "--path",
    required=True,
    help="Path to remove directory",
)
def remove_dir(team_id, path):
    from supervisely.cli.teamfiles import remove_dir
    try:
        success = remove_dir(team_id, path)
        if success:
            print(f"Directory '{path}' successfully removed")
            sys.exit(0)
        else:
            print(f"Removing directory failed")
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
    help="Supervisely team ID",
)
@click.option(
    "-s",
    "--src",
    required=True,
    help="Path to local source directory from which files are uploaded",
)
@click.option(
    "-d",
    "--dst",
    required=True,
    help="Path to teamfiles remote destination directory to which files are uploaded",
)
def upload(team_id, src, dst):
    from supervisely.cli.teamfiles import upload_to_teamfiles
    try:
        success = upload_to_teamfiles(team_id, src, dst)
        if success:
            print("Local directory uploaded to teamfiles sucessfully!")
            sys.exit(0)
        else:
            print("Upload failed")
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
    help="Supervisely team ID",
)
@click.option(
    "--task-id",
    required=True,
    help="Supervisely task ID",
)
@click.option(
    "-d",
    "--dir",
    required=True,
    help="Path to teamfiles directory",
)
def set_task_output_dir(team_id, task_id, dir):
    from supervisely.cli.teamfiles import set_task_output_dir
    try:
        success = set_task_output_dir(team_id, task_id, dir)
        if success:
            print("Setting task output directory succeed")
            sys.exit(0)
        else:
            print("Setting task output directory failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Aborting...")
        print("Setting task output directory aborted")
        sys.exit(1)



@cli.command(
    help="Get project name"
)
@click.option(
    "-id",
    "--id",
    required=True,
    help="Supervisely project ID",
)
def get_project_name(id):
    from supervisely.cli.env import get_project_name
    try:
        success = get_project_name(id)
        if success:
            sys.exit(0)
        else:
            print("Getting project name failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Getting project name directory aborted")
        sys.exit(1)

@cli.command(
    help="Get synced directory"
)
def get_synced_dir():
    from supervisely.cli.env import get_synced_dir
    try:
        success = get_synced_dir()
        if success:
            sys.exit(0)
        else:
            print("Getting synced directory failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Getting synced directory aborted")
        sys.exit(1)

