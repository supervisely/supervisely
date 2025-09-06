from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.app.widgets import (
    Container,
    Dialog,
    Field,
    GridGallery,
    SelectAppSession,
    Switch,
    Widget,
)
from supervisely.nn.model.model_api import ModelAPI
from supervisely.sly_logger import logger
from supervisely.solution.utils import find_agent


class PreLabelingGUI(Widget):
    PREDICT_APP_SLUG = "supervisely-ecosystem/apply-nn"

    def __init__(
        self,
        api: Optional[Api] = None,
        team_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        widget_id: Optional[str] = None,
    ):
        ##### input info #########
        self.api = api or Api.from_env()
        self.team_id = team_id or sly_env.team_id()
        self.workspace_id = workspace_id or sly_env.workspace_id()
        ##########################

        ##### settings ###########
        self._model = None
        self._classes = []
        self._session_id = None
        self._predict_app_task_id = None
        ##########################

        ###### temp results ######
        self._processed_images = []
        self._last_processed_images = []
        self._result = None
        ##########################

        self.content = self._init_gui()
        super().__init__(widget_id=widget_id)

    @property
    def modal(self) -> Dialog:
        if not hasattr(self, "_settings_modal"):
            self._settings_modal = Dialog(title="Pre-labeling Settings", content=self.content)
        return self._settings_modal

    @property
    def session_id(self) -> Optional[int]:
        """Get the current session ID."""
        return self._session_id

    @property
    def predict_app_task_id(self) -> Optional[int]:
        """Get the current predict app session."""
        self._predict_app_task_id = DataJson()[self.widget_id]["predict_app_task_id"]
        return self._predict_app_task_id

    @predict_app_task_id.setter
    def predict_app_task_id(self, task_id: Union[int, None]):
        """Set the predict app session task ID."""
        self._predict_app_task_id = task_id
        DataJson()[self.widget_id]["predict_app_task_id"] = task_id
        DataJson().send_changes()

    @property
    def model(self) -> Optional[ModelAPI]:
        """Get the currently deployed model."""
        if self._session_id is None:
            return None
        if not hasattr(self, "_model") or self._model is None:
            self._model = self.api.nn.connect(self._session_id)
        if self._model.task_id != self._session_id:
            self._model.shutdown()
            self._model = self.api.nn.connect(self._session_id)
        return self._model

    def _init_gui(self):
        # Enable/Disable switch
        enable_field = Field(
            self.enable_switch,
            title="Enable Pre-labeling",
            description="Enable or disable automatic pre-labeling of sampled images using the deployed custom model.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-settings",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        # Session ID selection
        session_id_field = Field(
            self.select_session,
            title="Model Session",
            description="Selected model session for pre-labeling.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-collection-item",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        # Reconnect to new model switch
        reconnect_field = Field(
            content=self.connect_to_new_switch,
            title="Reconnect to New Model",
            description="Switch on to automatically reconnect to a new model (e.g. after training and comparison of models).",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-settings",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        # Preview gallery section
        preview_field = Field(
            self.preview_gallery,
            title="Last Processed Images",
            description="View the last processed images by the pre-labeling model.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-image",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        return Container([enable_field, session_id_field, reconnect_field, preview_field], gap=20)

    @property
    def enable_switch(self) -> Switch:
        if not hasattr(self, "_enable_switch"):
            self._enable_switch = Switch(switched=False)
        return self._enable_switch

    @property
    def connect_to_new_switch(self) -> Switch:
        if not hasattr(self, "_connect_to_new_switch"):
            self._connect_to_new_switch = Switch(switched=True)
        return self._connect_to_new_switch

    @property
    def select_session(self) -> SelectAppSession:
        if not hasattr(self, "_select_session"):
            self._select_session = SelectAppSession(
                team_id=self.team_id, tags=["deployed_nn"], size="small"
            )
            # self._select_session.disable()
        return self._select_session

    @property
    def preview_gallery(self) -> GridGallery:
        if not hasattr(self, "_preview_gallery"):
            self._preview_gallery = GridGallery(columns_number=3)
        return self._preview_gallery

    def get_json_data(self) -> dict:
        return {
            "session_id": self._session_id,
            "enabled": self.enable_switch.is_switched(),
            "processed_images": self._get_processed_images(),
            "last_processed_images": self._get_last_processed_images(),
            "predict_app_task_id": self._predict_app_task_id,
        }

    def get_json_state(self) -> dict:
        return {}

    def save_settings(self, enabled: bool):
        """Save pre-labeling settings to DataJson."""
        DataJson()[self.widget_id]["settings"] = {
            "enabled": enabled,
        }
        DataJson().send_changes()

    def load_settings(self):
        """Load pre-labeling settings from DataJson."""
        data = DataJson().get(self.widget_id, {}).get("settings", {})
        enabled = data.get("enabled", True)
        self.update_widgets(enabled)

    def update_widgets(self, enabled: bool):
        """Update widgets based on settings."""
        if enabled:
            self.enable_switch.on()
        else:
            self.enable_switch.off()

    def _get_processed_images(self) -> List[int]:
        """Get processed images from DataJson."""
        return DataJson().get(self.widget_id, {}).get("processed_images", [])

    def add_processed_images(self, image_ids: List[int]):  # TODO: maybe dict [ds_id, List[int]]?
        """Add processed images to DataJson."""
        if "processed_images" not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id]["processed_images"] = []
        for image_id in image_ids:
            DataJson()[self.widget_id]["processed_images"].append(image_id)
        DataJson()[self.widget_id]["last_processed_images"] = deepcopy(image_ids)
        DataJson().send_changes()

    def _get_last_processed_images(self) -> List[int]:
        """Get last preview images from DataJson."""
        return DataJson().get(self.widget_id, {}).get("last_processed_images", [])

    def set_model_session_id(self, session_id: int):
        """Set the task ID for the model."""
        if not isinstance(session_id, int):
            raise ValueError("Session ID must be an integer.")
        elif session_id == self._session_id:
            return
        elif self.session_id is not None and not self._connect_to_new_switch.is_switched():
            return
        self._connect_model(session_id)
        self._session_id = session_id
        self.select_session.set_session_id(session_id)
        DataJson()[self.widget_id]["session_id"] = session_id
        DataJson().send_changes()

    def _connect_model(self, session_id: int):
        """Connect to the model using the session ID."""
        self.select_session.set_session_id(session_id)
        DataJson()[self.widget_id]["session_id"] = session_id
        DataJson().send_changes()

        if self._model is None:
            self._model = self.api.nn.connect(session_id)
            self._classes = self._get_model_classes()
        elif self._model.task_id != session_id:
            self._model.shutdown()
            self._model = self.api.nn.connect(session_id)
            self._classes = self._get_model_classes()
        elif not self._model.is_deployed():
            self._model = self.api.nn.connect(session_id)
            self._classes = self._get_model_classes()

        # if self.enable_switch.is_switched():
        #     self._run_predict_app()

    def _run_predict_app(self):
        """Ensure that the predict app session is set."""
        try:
            if self.predict_app_task_id and not self.api.task.is_running(self.predict_app_task_id):
                self.predict_app_task_id = None
            if self.predict_app_task_id is None:
                module_id = self.api.app.get_ecosystem_module_id(slug=self.PREDICT_APP_SLUG)
                agent_id = find_agent(self.api, self.team_id)
                session_info = self.api.app.start(
                    module_id=module_id,
                    workspace_id=self.workspace_id,
                    agent_id=agent_id,
                )

                self.api.app.wait_until_ready_for_api_calls(
                    session_info.task_id, attempts=100, attempt_delay_sec=5
                )
                self.predict_app_task_id = session_info.task_id
        except Exception as e:
            logger.error(f"Failed to prepare predict app session: {repr(e)}")
            self.predict_app_task_id = None
            self.enable_switch.off()

    def update_preview_gallery(self, images: List[int]):
        """Update preview gallery with new images."""
        self.preview_gallery.clean_up()

        if not images:
            logger.warning("No images to update in preview gallery.")
            return

        if not self.model:
            logger.warning("No model connected. Cannot update preview gallery.")
            return

        if not self.model.is_deployed():
            logger.warning("Model is not deployed. Cannot update preview gallery.")
            return

        # Limit to last 3 images for preview
        if len(images) > 3:
            images = images[-3:]

        image_infos = self.api.image.get_info_by_id_batch(images)
        urls_map = {img.id: img.full_storage_url for img in image_infos}

        for p in self.model.predict_detached(image_id=images):
            image_url = urls_map.get(p.image_id)
            self.preview_gallery.append(image_url=image_url, annotation=p.annotation)

    def run(
        self,
        images: List[int],
        confidence_threshold: float = 0.5,
        task_id: Optional[int] = None,
        model_session_id: Optional[int] = None,
        iou_merge_threshold: Optional[float] = 0.5,
    ):
        """Run pre-labeling on the provided images."""
        try:
            if not self.enable_switch.is_switched():
                raise ValueError("Pre-labeling is disabled. Enable it to run pre-labeling.")

            if model_session_id:
                self.set_model_session_id(model_session_id)

            if task_id:
                self.predict_app_task_id = task_id

            if self._session_id is None:
                raise ValueError("Please select a model session for pre-labeling.")

            if not self.model:
                raise ValueError("No model is connected. Please deploy a model first.")

            if not images:
                raise ValueError("No images available for pre-labeling.")

            self._run_predict_app()

            images_infos = self.api.image.get_info_by_id_batch(images)
            dataset_ids = list({img.dataset_id for img in images_infos})
            project_id = self.api.dataset.get_info_by_id(dataset_ids[0]).project_id

            # Process images with the model
            data = {
                "model": {"mode": "connect", "session_id": self._session_id},
                "input": {"image_ids": images, "project_id": project_id},
                "settings": {
                    "predictions_mode": "Merge with existing labels",
                    "inference_settings": {
                        "confidence_threshold": confidence_threshold,
                        "existing_objects_iou_threshold": 0.65,
                    },
                },
                "classes": self._classes,
                "output": {
                    "upload_to_source_project": True,
                },
            }
            if iou_merge_threshold is not None:
                data["settings"]["iou_merge_threshold"] = iou_merge_threshold
            logger.info(f"Running pre-labeling on {len(images)} images with settings: {data}")
            res = self.api.task.send_request(
                task_id=self.predict_app_task_id, data=data, method="predict", retries=1, timeout=30
            )
            if not res:
                logger.error("Pre-labeling failed or was skipped.")
                self._result = None
                # self._processed_images = []
                self._last_processed_images = []
                return None
            else:
                logger.info("Pre-labeling completed successfully.")
                processed_images = [pred["image_id"] for pred in res]
                self._result = res
                self._processed_images.extend(processed_images)
                self._last_processed_images = processed_images
                self.add_processed_images(processed_images)

        except Exception as e:
            logger.error(f"Failed to run pre-labeling: {repr(e)}")
            return
        finally:
            if self.predict_app_task_id:
                try:
                    # self.api.task.stop(self.predict_app_task_id)
                    self.predict_app_task_id = None
                except Exception as e:
                    logger.error(f"Failed to stop predict app session: {repr(e)}")

    def _get_model_classes(self) -> List[str]:
        """Get the classes from the connected model."""
        if not self.model:
            return []
        return self.model.get_classes()
