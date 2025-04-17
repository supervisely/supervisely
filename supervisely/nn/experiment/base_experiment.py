import json
import os
from typing import Any, Dict, Optional, Union
from pathlib import Path

import supervisely.io.env as sly_env
from supervisely.api.api import Api
import supervisely.io.fs as sly_fs
import supervisely.io.json as sly_json
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
        """
        self.info = experiment_info
        self.api = api
        self.hyperparameters = hyperparameters
        self.model_meta = ProjectMeta.from_json(model_meta)
        self.serving_class = serving_class
        self.team_id = sly_env.team_id()
        self.artifacts_dir = experiment_info.get("artifacts_dir")
   
    def generate_report(self) -> str:
        """Generate and upload experiment report to Supervisely"""
        generator = BaseExperimentGenerator(
            api=self.api,
            experiment_info=self.info,
            hyperparameters=self.hyperparameters,
            model_meta=self.model_meta,
            serving_class=self.serving_class,
        )
        generator.generate_report()