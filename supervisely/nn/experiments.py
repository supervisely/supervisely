from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from os.path import dirname, join
from typing import Any, Dict, List, NamedTuple

import requests

from supervisely import logger
from supervisely.api.api import Api, ApiField


class ExperimentInfo(NamedTuple):
    experiment_name: str
    """Name of the experiment. Defined by the user in the training app"""
    framework_name: str
    """Name of the framework used in the experiment"""
    model_name: str
    """Name of the model used in the experiment. Defined by the user in the training app"""
    task_type: str
    """Task type of the experiment"""
    project_id: int
    """Project ID in Supervisely"""
    task_id: int
    """Task ID in Supervisely"""
    model_files: Dict[str, str]
    """Dictionary with paths to model files that needs to be downloaded for training"""
    checkpoints: List[str]
    """List of relative paths to checkpoints"""
    best_checkpoint: str
    """Name of the best checkpoint. Defined by the user in the training app"""
    app_state: str
    """Path to file with settings that were used in the app"""
    model_meta: str
    """Path to file with model metadata such as model name, project id, project name and classes used for training"""
    train_val_split: str
    """Path to train and validation splits, which contains IDs of the images used in each split"""
    hyperparameters: str
    """Path to .yaml file with hyperparameters used in the experiment"""
    artifacts_dir: str
    """Path to the directory with artifacts"""
    datetime: str
    """Date and time when the experiment was started"""
    evaluation_report_id: int
    """ID of the evaluation report"""
    eval_metrics: Dict[str, Any]
    """Evaluation metrics"""


def get_experiment_infos(api: Api, team_id: int, framework_name: str) -> List[ExperimentInfo]:
    """
    Get experiments from the specified framework folder for Train v2

    :param api: Supervisely API client
    :type api: Api
    :param team_id: Team ID
    :type team_id: int
    :param framework_name: Name of the framework
    :type framework_name: str
    :return: List of ExperimentInfo objects
    :rtype: List[ExperimentInfo]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        api = sly.Api.from_env()
        team_id = sly.env.team_id()
        framework_name = "rt-detr"
        experiment_infos = sly.nn.training.experiments.get_experiment_infos(api, team_id, framework_name)
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
            required_fields = {field for field in ExperimentInfo._fields}
            if not required_fields.issubset(response_json.keys()):
                logger.debug(
                    f"Missing required fields in JSON from '{experiment_path}': {required_fields - response_json.keys()}"
                )
                return None
            return ExperimentInfo(**response_json)
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to fetch train metadata from '{experiment_path}': {e}")
        except JSONDecodeError as e:
            logger.debug(f"Failed to decode JSON from '{experiment_path}': {e}")
        return None

    with ThreadPoolExecutor() as executor:
        experiment_infos = list(executor.map(fetch_experiment_data, sorted_file_infos))

    experiment_infos = [info for info in experiment_infos if info is not None]
    return experiment_infos
