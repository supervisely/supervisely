import os
import random
import time
from typing import Any, Dict, List

import yaml

from supervisely._utils import logger
from supervisely.annotation.annotation import Annotation, Label
from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import (
    Button,
    Flexbox,
    Field,
    Container,
    Card,
    Text,
    Input,
    DeployModel,
    Editor,
    FastTable,
    GridGallery,
    InputNumber,
    OneOf,
    Progress,
    ProjectThumbnail,
    RadioGroup,
    RadioTable,
    SelectDataset,
    Stepper,
)
from supervisely.io import env
from supervisely.nn.model.model_api import ModelAPI
from supervisely.nn.model.prediction import Prediction
from supervisely.project.project import ProjectType
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_annotation import VideoAnnotation

from supervisely.nn.inference.predict_app.gui.input_selector import InputSelector


class PredictAppGui:

    def __init__(self, api: Api, static_dir: str = "static"):
        self.api = api
        self.static_dir = static_dir
        self.team_id = env.team_id()
        self.model = DeployModel(api=self.api, team_id=self.team_id)
        self.model.deploy = self._deploy_model

        self.model_card = Card(title="Select Model", description="", content=self.model)
        self.inference_settings = Editor("", language_mode="yaml", height_px=300)
        self._stop_flag = False
        self._is_running = False

        # Steps
        self.input_selector = InputSelector(self.api)
        self.model_selector = ModelSelector(self.api)
        self.classes_selector = ClassesSelector(self.api)
        self.tags_selector = TagsSelector(self.api)
        self.settings_selector = SettingsSelector(self.api)
        self.preview = Preview()
        self.output_selector = OutputSelector(self.api)
        # -------------------------------- #

        # Stepper
        step_titles = [
            self.input_selector.title,
            self.model_selector.title,
            self.classes_selector.title,
            self.tags_selector.title,
            self.settings_selector.title,
            self.preview.title,
            self.output_selector.title,
        ]
        step_widgets = [
            self.input_selector.card,
            self.model_selector.card,
            self.classes_selector.card,
            self.tags_selector.card,
            self.settings_selector.card,
            self.preview.card,
            self.output_selector.card,
        ]
        self.stepper = Stepper(titles=step_titles, widgets=step_widgets)
        # ---------------------------- #
        self.layout = Container([self.stepper])

        @self.output.run_button.click
        def run_button_click():
            self.run()

    def run(self, run_parameters: Dict[str, Any] = None) -> List[Prediction]:
        self.output.validation_message.hide()
        if run_parameters is None:
            run_parameters = self.get_run_parameters()

        if self.model.model_api is None:
            self.model._deploy()

        model_api = self.model.model_api
        if model_api is None:
            logger.error("Model Deployed with an error")
            return

        inference_settings = run_parameters["inference_settings"]
        if not inference_settings:
            inference_settings = {}

        item_prameters = run_parameters["item"]

        output_parameters = run_parameters["output"]
        upload_parameters = {}
        upload_mode = output_parameters["mode"]

        upload_parameters["upload_mode"] = upload_mode
        if upload_mode == "iou_merge":
            upload_parameters["existing_objects_iou_thresh"] = output_parameters[
                "iou_merge_threshold"
            ]

        if upload_mode == "create":
            project_name = output_parameters["project_name"]
            if not project_name:
                self.output.validation_message.set(
                    "Project name cannot be empty when creating a new project.", "error"
                )
                self.output.validation_message.show()
                return
            created_project = self.api.project.create(
                env.workspace_id(),
                project_name,
                type=ProjectType.IMAGES,
                change_name_if_conflict=True,
            )
            upload_parameters["output_project_id"] = created_project.id
            upload_parameters["upload_mode"] = "append"

        predictions = []
        self._is_running = True
        try:
            with model_api.predict_detached(
                **item_prameters,
                **inference_settings,
                **upload_parameters,
                tqdm=self.output.progress(),
            ) as session:
                i = 0
                for prediction in session:
                    if "output_project_id" in upload_parameters:
                        prediction.extra_data["output_project_id"] = upload_parameters[
                            "output_project_id"
                        ]
                    if i % 100 == 0:
                        if "output_project_id" in prediction.extra_data:
                            project_id = prediction.extra_data["output_project_id"]
                        else:
                            project_id = prediction.project_id
                        self.output.set_result_thumbnail(project_id)
                    predictions.append(prediction)
                    i += 1
                    if self._stop_flag:
                        logger.info("Prediction stopped by user.")
                        break
        finally:
            self._is_running = False
            self._stop_flag = False

        return predictions

    def stop(self):
        logger.info("Stopping prediction...")
        self._stop_flag = True

    def wait_for_stop(self, timeout: int = None):
        logger.info(
            "Waiting " + ""
            if timeout is None
            else f"{timeout} seconds " + "for prediction to stop..."
        )
        t = time.monotonic()
        while self._is_running:
            if timeout is not None and time.monotonic() - t > timeout:
                raise TimeoutError("Timeout while waiting for stop.")
            time.sleep(0.1)
        logger.info("Prediction stopped.")

    def shutdown_model(self):
        self.stop()
        self.wait_for_stop(10)
        self.model.stop()

    def _deploy_model(self) -> ModelAPI:
        model_api = None
        self.preview.card.unlock()
        self.input_selector.card.unlock()
        try:
            model_api = type(self.model).deploy(self.model)
            inference_settings = model_api.get_settings()
            self.set_inference_settings(inference_settings)
        except:
            self.output.run_button.disable()
            self.preview.preview_button.disable()
            self.preview.card.lock("Deploy model first to preview results.")
            self.input_selector.card.lock("Deploy model first to select items.")
            self.set_inference_settings("")
            raise
        else:
            self.preview.preview_button.enable()
            self.output.run_button.enable()
        return model_api

    def get_inference_settings(self):
        return yaml.safe_load(self.inference_settings.get_text())

    def set_inference_settings(self, settings: Dict[str, Any]):
        if isinstance(settings, str):
            self.inference_settings.set_text(settings)
        else:
            self.inference_settings.set_text(yaml.safe_dump(settings))

    def get_run_parameters(self) -> Dict[str, Any]:
        return {
            "model": self.model.get_deploy_parameters(),
            "inference_settings": self.get_inference_settings(),
            "item": self.input_selector.get_input_settings(),
            "output": self.output.get_output_settings(),
        }

    def load_from_json(self, data):
        self.model.load_from_json(data.get("model", {}))
        inference_settings = data.get("inference_settings", "")
        self.set_inference_settings(inference_settings)
        self.input_selector.load_from_json(data.get("items", {}))
        self.output.load_from_json(data.get("output", {}))
