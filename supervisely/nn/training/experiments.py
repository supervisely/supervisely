from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from os.path import dirname, join
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
    train_val_splits: dict
    app_state: dict
    datetime: str
    evaluation_report_id: int
    eval_metrics: Dict[str, Any]


def get_experiment_infos(api: Api, team_id: int, framework_name: str) -> List[ExperimentInfo]:
    """
    Get experiments from the specified framework folder for Train v2
    """
    metadata_name = "experiment_info.json"
    experiments_folder = "/experiments"
    experiment_infos = []

    file_infos = api.file.list(team_id, experiments_folder, recursive=True, return_type="fileinfo")
    sorted_file_infos = []
    for file_info in file_infos:
        if not file_info.path.endswith(metadata_name):
            continue

        experiment_dir = dirname(file_info.path)
        if experiment_dir.endswith(framework_name):
            experiment_path = join(experiment_dir, metadata_name)
            sorted_file_infos.append(experiment_path)

    def fetch_experiment_data(file_info):
        try:
            response = api.post(
                "file-storage.download",
                {ApiField.TEAM_ID: team_id, ApiField.PATH: file_info},
                stream=True,
            )
            response.raise_for_status()
            response_json = response.json()
            return ExperimentInfo(**response_json)
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to fetch train metadata from '{experiment_path}': {e}")
        except JSONDecodeError as e:
            logger.debug(f"Failed to decode JSON from '{experiment_path}': {e}")
        return None
    
    with ThreadPoolExecutor() as executor:
        experiment_infos = list(executor.map(fetch_experiment_data, sorted_file_infos))

    return experiment_infos
