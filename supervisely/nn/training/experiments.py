from concurrent.futures import ThreadPoolExecutor, as_completed
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


def get_experiment_infos(api: Api, team_id: int, framework_name: str) -> List[ExperimentInfo]:
    """
    Get experiments from the specified framework folder for Train v2
    """
    metadata_name = "experiment_info.json"
    experiments_folder = "/experiments"
    experiment_infos = []

    file_infos = api.file.list(team_id, experiments_folder, recursive=True, return_type="fileinfo")

    def fetch_experiment_data(file_info):
        if not file_info.path.endswith(framework_name):
            return None

        experiment_path = join(file_info.path, metadata_name)
        try:
            response = api.post(
                "file-storage.download",
                {ApiField.TEAM_ID: team_id, ApiField.PATH: experiment_path},
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
        futures = [executor.submit(fetch_experiment_data, file_info) for file_info in file_infos]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                experiment_infos.append(result)

    return experiment_infos
