import json
import os
from typing import Any, Dict, Optional, Union

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.io.fs import mkdir
from supervisely.nn.experiment.base_experiment_generator import BaseExperimentGenerator
from supervisely.nn.inference import Inference
from supervisely.project import ProjectMeta


class BaseExperiment:
    """
    Base class for working with experiments.

    Provides functionality for managing experiment information and artifacts.
    """

    def __init__(
        self,
        api: Api,
        experiment_info: Dict[str, Any],
        hyperparameters: Optional[Union[str, Dict]] = None,
        model_meta: Optional[Union[str, Dict]] = None,
        serving_class: Optional[Inference] = None,
        output_dir: str = "./experiment_report",
    ):
        """Initialize experiment class.

        :param api: Supervisely API client
        :type api: Api
        :param experiment_info: Dictionary with experiment information
        :type experiment_info: Dict[str, Any]
        :param hyperparameters: Hyperparameters as YAML string or dictionary
        :type hyperparameters: Optional[Union[str, Dict]]
        :param model_meta: Model metadata as dictionary
        :type model_meta: Optional[Union[str, Dict]]
        :param output_dir: Directory to save the report
        :type output_dir: str
        """
        self.info = experiment_info
        self.api = api
        self.hyperparameters = hyperparameters
        self.model_meta = ProjectMeta.from_json(model_meta)
        self.serving_class = serving_class
        self.output_dir = output_dir
        self.team_id = sly_env.team_id()
        self.artifacts_dir = experiment_info.get("artifacts_dir")

    def generate_report(self) -> str:
        """Generate experiment report and return path to README.md.

        :return: Path to the generated README.md file
        :rtype: str
        """
        generator = BaseExperimentGenerator(
            experiment_info=self.info,
            hyperparameters=self.hyperparameters,
            model_meta=self.model_meta,
            serving_class=self.serving_class,
            output_dir=self.output_dir,
        )
        return generator.generate_report()
