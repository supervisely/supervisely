import os
import shlex
import shutil
import subprocess
import tempfile
import atexit
from pathlib import Path
from typing import List, Optional

from dotenv import dotenv_values, load_dotenv
from rich.console import Console


DEFAULT_IMPORT_IMAGE = "supervisely/main-import-cli:latest"
DEFAULT_ENV_FILE = "~/supervisely.env"
_TEMP_ENV_FILES: List[str] = []


def _cleanup_temp_env_files():
    for path in _TEMP_ENV_FILES:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


atexit.register(_cleanup_temp_env_files)


def _has_credentials() -> bool:
    return bool(os.environ.get("SERVER_ADDRESS")) and bool(os.environ.get("API_TOKEN"))


def _docker_env_file_from_values(values: dict) -> Optional[List[str]]:
    server_address = values.get("SERVER_ADDRESS")
    api_token = values.get("API_TOKEN")
    if not server_address or not api_token:
        return None

    fd, path = tempfile.mkstemp(prefix="sly-import-env-", suffix=".env")
    os.close(fd)
    os.chmod(path, 0o600)
    with open(path, "w") as env_file:
        env_file.write(f"SERVER_ADDRESS={server_address}\n")
        env_file.write(f"API_TOKEN={api_token}\n")
    _TEMP_ENV_FILES.append(path)
    return ["--env-file", path]


def _prepare_env_args(console: Console, env_file: Optional[str]) -> Optional[List[str]]:
    if env_file is None or env_file == "":
        if not _has_credentials():
            console.print(
                "\nError: SERVER_ADDRESS and API_TOKEN are not set. "
                "Provide ~/supervisely.env, --env-file, or environment variables.\n",
                style="bold red",
            )
            return None
        return ["-e", "SERVER_ADDRESS", "-e", "API_TOKEN"]

    env_path = Path(env_file).expanduser()
    if env_path.exists():
        load_dotenv(env_path)
        env_args = _docker_env_file_from_values(dotenv_values(env_path))
        if env_args is not None:
            return env_args

        console.print(
            f"\nError: Env file '{env_path}' doesn't contain SERVER_ADDRESS and API_TOKEN.\n",
            style="bold red",
        )
        return None

    if _has_credentials():
        return ["-e", "SERVER_ADDRESS", "-e", "API_TOKEN"]

    console.print(
        f"\nError: Env file '{env_path}' doesn't exist and SERVER_ADDRESS/API_TOKEN "
        "are not set in the current environment.\n",
        style="bold red",
    )
    return None


def build_docker_command(
    src: str,
    project_id: int,
    dataset_id: Optional[int] = None,
    dataset_name: Optional[str] = None,
    import_as_links: bool = False,
    image: str = DEFAULT_IMPORT_IMAGE,
    env_file: Optional[str] = DEFAULT_ENV_FILE,
) -> Optional[List[str]]:
    console = Console()
    src_path = Path(src).expanduser().resolve()

    if not src_path.exists():
        console.print(f"\nError: Source path '{src_path}' doesn't exist\n", style="bold red")
        return None

    env_args = _prepare_env_args(console, env_file)
    if env_args is None:
        return None

    if src_path.is_dir():
        mount_source = src_path
        container_input = "/input"
    else:
        mount_source = src_path.parent
        container_input = f"/input/{src_path.name}"

    command = [
        "docker",
        "run",
        "--rm",
        *env_args,
        "-e",
        f"PROJECT_ID={project_id}",
        "-e",
        "SLY_APP_DATA_DIR=/tmp/sly-import-work",
        "-v",
        f"{mount_source}:/input:ro",
    ]

    if dataset_id is not None:
        command.extend(["-e", f"DATASET_ID={dataset_id}"])
    if dataset_name:
        command.extend(["-e", f"DATASET_NAME={dataset_name}"])
    if import_as_links:
        command.extend(["-e", "IMPORT_AS_LINKS=true"])

    command.extend([image, "--input", container_input])
    return command


def import_run(
    src: str,
    project_id: int,
    dataset_id: Optional[int] = None,
    dataset_name: Optional[str] = None,
    import_as_links: bool = False,
    image: str = DEFAULT_IMPORT_IMAGE,
    env_file: Optional[str] = DEFAULT_ENV_FILE,
    dry_run: bool = False,
) -> bool:
    console = Console()

    command = build_docker_command(
        src=src,
        project_id=project_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        import_as_links=import_as_links,
        image=image,
        env_file=env_file,
    )
    if command is None:
        return False

    if dry_run:
        console.print(shlex.join(command))
        return True

    if shutil.which("docker") is None:
        console.print(
            "\nError: Docker is not installed or is not available in PATH.\n",
            style="bold red",
        )
        return False

    try:
        completed = subprocess.run(command)
    except KeyboardInterrupt:
        console.print("\nImport aborted\n", style="bold red")
        return False
    except Exception as exc:
        console.print(f"\nError: Failed to run Docker: {exc}\n", style="bold red")
        return False

    return completed.returncode == 0
