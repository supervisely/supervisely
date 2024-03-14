from os.path import join
from typing import List

from supervisely.api.api import Api
from supervisely.io.fs import silent_remove
from supervisely.nn.checkpoints.checkpoint import CheckpointInfo


def get_list(api: Api, team_id: int) -> List[CheckpointInfo]:
    checkpoints = []
    training_app_directory = "/mmdetection-3/"
    if not api.file.dir_exists(team_id, training_app_directory):
        return []
    task_files_infos = api.file.list(team_id, training_app_directory, recursive=False)
    for task_file_info in task_files_infos:
        task_id = task_file_info["name"].split("_")[0]
        session_dir_files = api.file.list(team_id, task_file_info["path"], recursive=False)
        session_link = f"{api.server_address}/apps/sessions/{task_id}"
        checkpoints_infos = [file for file in session_dir_files if file["name"].endswith(".pth")]
        if not api.file.exists(team_id, join(task_file_info["path"], "config.py")):
            continue
        api.file.download(team_id, join(task_file_info["path"], "config.py"), "model_config.txt")
        with open("model_config.txt", "r") as f:
            lines = f.readlines()
            project_line = lines[-1] if lines else None
            start = project_line.find("'") + 1
            end = project_line.find("'", start)
            project_name = project_line[start:end]
            task_type_line = lines[-3] if lines else None
            start = task_type_line.find("'") + 1
            end = task_type_line.find("'", start)
            task_type = task_type_line[start:end].replace("_", " ")
            f.close()
        silent_remove("model_config.txt")
        if len(checkpoints_infos) == 0:
            continue
        checkpoint_info = CheckpointInfo(
            app_name="Train MMDetection 3.0",
            session_id=task_id,
            session_path=task_file_info["path"],
            session_link=session_link,
            task_type=task_type,
            training_project_name=project_name,
            checkpoints=checkpoints_infos,
        )
        checkpoints.append(checkpoint_info)
    return checkpoints
