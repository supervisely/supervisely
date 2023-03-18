import click


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
    import sys
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


# training tensorboard template
@cli.command(
    help="Download project data from supervisely to local directory"
)
@click.option(
    "--project-id",
    required=True,
    help="Supervisely project ID",
)
@click.option(
    "--save-dir",
    required=True,
    help="Save directory",
)
def download_project(project_id, save_dir):
    from supervisely.cli.project import download_project
    import sys
    try:
        success = download_project(
            project_id = project_id,
            save_dir = save_dir,
        )
        if success:
            print("Project downloaded sucessfully!")
            sys.exit(0)
        else:
            print(f"Project not downloaded with following error: {success[1]}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Aborting...")
        print("Project not downloaded")
        sys.exit(1)


@cli.command(
    help="Upload local files to supervisely teamfiles"
)
@click.option(
    "--team-id",
    required=True,
    help="Supervisely team ID",
)
@click.option(
    "--from-local-dir",
    required=True,
    help="Path to local directory from which files are uploaded",
)
@click.option(
    "--to-teamfiles-dir",
    required=True,
    help="Path to teamfiles directory to which files are uploaded",
)
def upload_to_teamfiles(team_id, from_local_dir, to_teamfiles_dir):
    from supervisely.cli.teamfiles import upload_to_teamfiles
    import sys
    try:
        success = upload_to_teamfiles(
            team_id = team_id,
            from_local_dir = from_local_dir,
            to_teamfiles_dir = to_teamfiles_dir
        )
        if success:
            print("Local directory uploaded to teamfiles sucessfully!")
            sys.exit(0)
        else:
            print("Upload failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Aborting...")
        print("Upload aborted")
        sys.exit(1)

@cli.command(
    help="Set link to teamfiles directory at workspace tasks interface"
)
@click.option(
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
    "--teamfiles-dir",
    required=True,
    help="Path to teamfiles directory",
)
def set_task_output_dir(team_id, task_id, teamfiles_dir):
    from supervisely.cli.teamfiles import set_task_output_dir
    import sys
    try:
        success = set_task_output_dir(
            team_id = team_id,
            task_id = task_id,
            teamfiles_dir = teamfiles_dir,
        )
        if success:
            print("Setting task output directory succeed")
            sys.exit(0)
        else:
            print(f"Setting task output directory failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("Aborting...")
        print("Setting task output directory aborted")
        sys.exit(1)