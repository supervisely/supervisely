import json
import os
import click
import re
import subprocess
import git
from dotenv import load_dotenv
from rich.console import Console

from supervisely.release.sly_release import (
    find_tag_in_repo,
    push_tag,
    get_app_from_instance,
    get_module_root,
    get_module_path,
    get_appKey,
    release,
    slug_is_valid,
    delete_tag,
)


def _check_git(repo: git.Repo):
    console = Console()
    result = True

    if len(repo.untracked_files) > 0:
        console.print(
            "[red][Error][/] You have untracked files. Commit all changes before releasing the app."
        )
        console.print("  Untracked files:")
        for i, file in enumerate(repo.untracked_files):
            console.print(f"  {i+1}) " + file)
        print()
        result = False

    if len(repo.index.diff(None)) > 0:
        console.print(
            "[red][Error][/] You have modified files. Commit all changes before releasing the app."
        )
        console.print("  Modified files:")
        console.print(
            "\n\n".join(
                f"  {i+1}) " + str(d).replace("\n", "\n  ")
                for i, d in enumerate(repo.index.diff(None))
            )
        )
        print()
        result = False

    if len(repo.index.diff("HEAD")) > 0:
        console.print(
            "[red][Error][/] You have staged files. Commit all changes before releasing the app."
        )
        console.print("  Staged files:")
        console.print(
            "\n\n".join(
                f"  {i+1}) " + str(d).replace("\n", "\n  ")
                for i, d in enumerate(repo.index.diff("HEAD"))
            )
        )
        print()
        result = False

    local_branch = repo.active_branch
    remote_branch = local_branch.tracking_branch()
    if remote_branch is None:
        console.print(
            "[red][Error][/] Your branch is not tracking any remote branches. Try running 'git push --set-upstream origin'\n"
        )
        result = False
    elif not local_branch.commit.hexsha == remote_branch.commit.hexsha:
        console.print(
            "[red][Error][/] Local branch and remote branch are different. Push your changes to a remote branch before releasing the app\n"
        )
        result = False

    return result


def _ask_release(repo: git.Repo):
    console = Console()
    if repo.active_branch.name in ("main", "master"):
        release_name = console.input(
            "Enter release name:\n",
        )
        try:
            sly_release_tags = [
                tag.name
                for tag in repo.tags
                if re.match("^sly-release-v\d+\.\d+\.\d+$", tag.name)
            ]
            sly_release_tags.sort(
                key=lambda tag: [
                    int(n) for n in tag.split("sly-release-v")[-1].split(".")
                ]
            )
            current_release_version = sly_release_tags[-1][13:]
            suggested_release_version = ".".join(
                [
                    *current_release_version.split(".")[:-1],
                    str(int(current_release_version.split(".")[-1]) + 1),
                ]
            )
        except Exception:
            current_release_version = None
            suggested_release_version = "0.0.1"
        while True:
            input_msg = f'Enter release version in format vX.X.X ({f"Last release: [blue]{current_release_version}[/]. " if current_release_version else ""}Press "Enter" for [blue]{suggested_release_version}[/]):\n'
            release_version = console.input(input_msg)
            if release_version == "":
                release_version = "v" + suggested_release_version
                break
            if re.fullmatch("v\d+\.\d+\.\d+", release_version):
                break
            console.print("Wrong version format. Should be of format vX.X.X")
    else:
        release_name = console.input("Enter release name:\n")
        console.print(
            f'Release version will be "{repo.active_branch.name}" as the branch name'
        )
        release_version = repo.active_branch.name
    return release_name, release_version


def _ask_confirmation(
    module_name, module_exists_label, release_version, release_name, server, branch_name
):
    console = Console()
    console.print()
    console.print(
        f'Application "{module_name}"([blue]{release_version}[/] "{release_name}") will be {module_exists_label} at "{server}" Supervisely instance.'
    )
    if branch_name in ["main", "master"]:
        console.print(
            f'Git tag "sly-release-{release_version}" will be added and pushed to origin'
        )
    while True:
        confirmed = input("Confirm? [y/n]:\n")
        if confirmed.lower() in ["y", "yes"]:
            return True
        if confirmed.lower() in ["n", "not"]:
            return False


def _run(app_directory, sub_app_directory, slug):
    console = Console()
    console.print("\nSupervisely Release App\n", style="bold")

    # 0) get variables
    if slug is not None and not slug_is_valid(slug):
        console.print("[red][Error][/] Invalid slug")
        return False
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    module_root = get_module_root(app_directory)
    module_path = get_module_path(module_root, sub_app_directory)
    console.print(f"Application directory:\t[green]{module_path}[/]")
    server_address = os.getenv("SERVER_ADDRESS", None)
    if server_address is None:
        console.print(
            '[red][Error][/] Cannot find [green]SERVER_ADDRESS[/]. Add it to your "~/supervisely.env" file or to environment variables'
        )
        return False
    console.print(f"Server address:\t\t[green]{server_address}[/]")
    api_token = os.getenv("API_TOKEN", None)
    if api_token is None:
        console.print(
            '[red][Error][/] Cannot find [green]API_TOKEN[/]. Add it to your "~/supervisely.env" file or to environment variables'
        )
        return False
    console.print(f"Api token:\t\t[green]{api_token[:4]}*******{api_token[-4:]}[/]")
    try:
        with open(module_path.joinpath("config.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        console.print(
            f'[red][Error][/] Cannot find "config.json" file at "{module_path}"'
        )
        return False
    except json.JSONDecodeError as e:
        console.print(
            f'[red][red][Error][/][/] Cannot decode config json file at "{module_path.joinpath("config.json")}": {str(e)}'
        )
        return False
    module_name = config.get("name", None)
    if module_name is None:
        console.print(
            f'[red][Error][/] Missing "name" field in config json file at "{module_path.joinpath("config.json")}"'
        )
        return False

    # 1) get git repo
    try:
        repo = git.Repo(module_root)
    except git.InvalidGitRepositoryError:
        console.print(
            f"[red][Error][/] [green]{module_path}[/] is not a git repository"
        )
        return False
    console.print(f"git branch:\t\t[green]{repo.active_branch}[/]")

    # 2) check that everything is commited and pushed
    result = _check_git(repo)
    if not result:
        return False

    # 3) get modal template
    modal_template = ""
    if "modal_template" in config:
        modal_template_path = module_root.joinpath(config["modal_template"])
        if not modal_template_path.exists() or not modal_template_path.is_file():
            console.print(
                f'[red][Error][/] Cannot find Modal Template at "{modal_template_path}". Please check your [green]config.json[/] file'
            )
            return False
        with open(modal_template_path, "r") as f:
            modal_template = f.read()

    # 4) get appKey
    appKey = get_appKey(repo, sub_app_directory)

    # 5) get app from instance ecosystem.info
    module_exists_label = "[yellow bold]updated[/]"
    try:
        app_data = get_app_from_instance(appKey, api_token, server_address)
        if app_data is None:
            module_exists_label = "[green bold]created[/]"
    except PermissionError:
        console.print(
            "[red][Error][/] Permission denied. Check that all credentials are set correctly.\n"
        )
        return False

    # 6) get release name and version from user
    release_name, release_version = _ask_release(repo)

    # 7) confirm
    confirmed = _ask_confirmation(
        module_name,
        module_exists_label,
        release_version,
        release_name,
        server_address,
        repo.active_branch.name,
    )
    if not confirmed:
        console.print("Release cancelled")
        return False

    # 8) add tag and push
    tag_created = False
    if repo.active_branch.name in ["main", "master"]:
        tag_name = f"sly-release-{release_version}"
        tag = find_tag_in_repo(tag_name, repo)
        if tag is None:
            tag_created = True
            repo.create_tag(tag_name)
        try:
            push_tag(tag_name, repo)
        except subprocess.CalledProcessError:
            if tag_created:
                repo.delete_tag(tag)
            console.print(
                f"[red][Error][/] Git push unsuccessful. You need write permissions in repository to release the application"
            )
            return False

    # 9) release
    console.print("Uploading archive...")
    success = True
    response = release(
        server_address,
        api_token,
        appKey,
        repo,
        config,
        release_name,
        release_version,
        modal_template,
        slug,
    )
    if response.status_code != 200:
        error = f"[red][Error][/] Error releasing the application. Please contact Supervisely team. Status Code: {response.status_code}"
        try:
            error += ", " + json.dumps(response.json())
        except:
            pass
        console.print(error)
        success = False
    elif response.json()["success"] != True:
        console.print(
            "[red][Error][/] Error releasing the application. Please contact Supervisely team"
            + json.dumps(response.json())
        )
        success = False

    if not success:
        if tag_created:
            console.print(f'Deleting tag "sly-release-{release_version}" from remote')
            try:
                delete_tag(f"sly-release-{release_version}", repo)
            except subprocess.CalledProcessError:
                console.print(
                    f'[orange1][Warning][/] Could not delete tag "sly-release-{release_version}" from remote. Please do it manually to avoid errors'
                )
        return False

    return True


# def cli_run():
#     parser = argparse.ArgumentParser(
#         description="This app allows you to release your aplication to supervisely platform",
#     )
#     parser.add_argument(
#         "-p", "--path", required=False, help="Path to the directory with application"
#     )
#     parser.add_argument(
#         "-a",
#         "--sub-app",
#         required=False,
#         help="Path to subApp relative to application directory",
#     )
#     parser.add_argument("-s", "--slug", required=False)
#     args = parser.parse_args()
#     console = Console()
#     try:
#         success = _run(
#             app_directory=args.path, sub_app_directory=args.sub_app, slug=args.slug
#         )
#         console.print("App released sucessfully!" if success else "App not released")
#     except KeyboardInterrupt:
#         console.print("Aborting...")
#         console.print("App not released")

@click.command(help="This app allows you to release your aplication to Supervisely platform")
@click.option("-p", "--path", required=False, help="[Optional] Path to the directory with application")
@click.option("-a", "--sub-app", required=False, help="[Optional] Path to sub-app relative to application directory")
@click.option("-s", "--slug", required=False, help="[Optional]")
def cli_run(path, sub_app, slug):
    console = Console()
    try:
        success = _run(
            app_directory=path, sub_app_directory=sub_app, slug=slug
        )
        console.print("App released sucessfully!" if success else "App not released")
    except KeyboardInterrupt:
        console.print("Aborting...")
        console.print("App not released")
