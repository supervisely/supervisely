from os.path import basename, join
from typing import List

from supervisely.api.api import Api
from supervisely.io.fs import silent_remove
from supervisely.io.json import load_json_file
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


def get_list(api: Api, team_id: int) -> List[CheckpointInfo]:
    checkpoints = []
    weights_dir_name = "checkpoints"
    training_app_directory = "/unet/"
    task_type = "instance segmentation"

    if not api.file.dir_exists(team_id, training_app_directory):
        return []

    task_files_infos = api.file.list(team_id, training_app_directory, recursive=False)
    for task_file_info in task_files_infos:
        task_id = task_file_info["name"].split("_")[0]
        session_dir_files = api.file.list(
            team_id, join(task_file_info["path"], weights_dir_name), recursive=False
        )
        session_link = f"{api.server_address}/apps/sessions/{task_id}"
        checkpoints_infos = [file for file in session_dir_files if file["name"].endswith(".pth")]
        config_path = join(task_file_info["path"], weights_dir_name, "train_args.json")
        if not api.file.exists(team_id, config_path):
            continue
        api.file.download(team_id, config_path, "model_config.json")
        config = load_json_file("model_config.json")
        project_name = basename(config["project_dir"].split("_")[0])
        silent_remove("model_config.json")
        if len(checkpoints_infos) == 0:
            continue
        checkpoint_info = CheckpointInfo(
            app_name="Train Unet",
            session_id=task_id,
            session_path=task_file_info["path"],
            session_link=session_link,
            task_type=task_type,
            training_project_name=project_name,
            checkpoints=checkpoints_infos,
        )
        checkpoints.append(checkpoint_info)
    return checkpoints
