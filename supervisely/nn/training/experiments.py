from json import JSONDecodeError
from os.path import join
from typing import Any, Dict, List, NamedTuple

import requests
from supervisely import logger
from supervisely.api.api import Api, ApiField


class ExperimentInfo(NamedTuple):
    model_name: str
    task_type: str
    model_files: Dict[str, str]  # {"config": "custom.yml"}
    checkpoints: List[str]
    best_checkpoint: str
    framework_name: str
    hyperparameters: Dict[str, Any]
    artifacts_dir: str
    task_id: int
    project_id: int
    train_dataset_id: int
    val_dataset_id: int
    datetime: str
    evaluation_report_id: int
    eval_metrics: Dict[str, Any]


def get_experiment_infos(
    api: Api, team_id: int, framework_folder: str
) -> List[ExperimentInfo]:
    """
    Get experiments from the specified framework folder for Train v2
    """
    metadata_name = "experiment_info.json"
    experiments_folder = join("/experiments", f"{framework_folder}/")
    file_infos = api.file.list(
        team_id, experiments_folder, recursive=True, return_type="fileinfo"
    )
    experiment_infos = []
    for file in file_infos:
        if file.name.endswith(metadata_name):
            try:
                response = api.post(
                    "file-storage.download",
                    {ApiField.TEAM_ID: team_id, ApiField.PATH: file.path},
                    stream=True,
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to fetch train metadata from '{file.path}': {e}")
                continue

            try:
                response_json = response.json()
                experiment_infos.append(ExperimentInfo(**response_json))
            except JSONDecodeError as e:
                logger.debug(f"Failed to decode JSON from '{file.path}': {e}")
                continue

    return experiment_infos
