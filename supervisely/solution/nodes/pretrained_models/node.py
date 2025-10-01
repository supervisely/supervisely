import re
import threading
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.api.entities_collection_api import CollectionTypeFilter
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    MoveLabeledDataFinishedMessage,
    TrainFinishedMessage,
)
from supervisely.solution.nodes.pretrained_models.automation import PretrainedModelsAuto
from supervisely.solution.nodes.pretrained_models.gui import PretrainedModelsGUI
from supervisely.solution.nodes.pretrained_models.history import (
    PretrainedModelsTasksHistory,
)

from supervisely.nn.inference import RuntimeType


class BaseTrainNode(BaseCardNode):
    PROGRESS_BADGE_KEY = "Training"
    TITLE = "Train Model"
    DESCRIPTION = "Train a custom model using the selected dataset and predefined configurations."
    ICON = "mdi mdi-robot"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        project_id: int,
        *args,
        **kwargs,
    ):

        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(self.project_id)

        # --- core blocks --------------------------------------------------------
        self.automation = PretrainedModelsAuto()
        self.gui = PretrainedModelsGUI(self.api, self.project)
        self.modal_content = self.gui.widget  # for BaseCardNode
        self.history = PretrainedModelsTasksHistory(self.api)

        # --- training settings --------------------------------------------------
        self._train_settings = None

        # --- node init ----------------------------------------------------------
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        # First, initialize the base class (to wrap publish/subscribe methods)
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        # --- NewExperiment widget ----------------------------------------------------------
        self._extra_widgets = [self.gui.widget]

        @self.click
        def on_click():
            self.gui.widget.visible = True
            self.gui.train_val_split_mode = "collections"
            train_collections, val_collections = self.gui._get_train_val_collections()
            self.gui.widget.train_collections = train_collections
            self.gui.widget.val_collections = val_collections

        @self.gui.widget.app_started
        def _on_app_started(app_id: int, model_id: int, task_id: int):
            self.gui.widget.visible = False
            self._previous_task_id = task_id
            self.run(app_id, model_id, task_id)

        @self.automation.apply_button.click
        def on_automate_click():
            self.automation.modal.hide()
            self.apply_automation()

    def configure_automation(self, *args, **kwargs):
        self.automation.func = self.run

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "data_versioning_project_id",
                "type": "source",
                "position": "top",
                "label": "Data Versioning",
                "connectable": True,
            },
            {
                "id": "training_finished",
                "type": "source",
                "position": "bottom",
                "label": "Output",
                "connectable": True,
            },
            {
                "id": "register_experiment",
                "type": "source",
                "position": "right",
                "label": "Output",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "register_experiment": self._send_training_finished_message,
            "training_finished": self._send_training_output_message,
        }

    def _available_subscribe_methods(self):
        """Returns a dictionary of methods that can be used as callbacks for subscribed events."""
        return {"data_versioning_project_id": self._get_data_versioning_project_id}

    def _get_data_versioning_project_id(self) -> Optional[int]:
        return getattr(self, "project_id", None)

    def _send_training_output_message(
        self, success: bool, task_id: int, experiment_info: dict
    ) -> TrainFinishedMessage:
        return TrainFinishedMessage(
            success=success,
            task_id=task_id,
            experiment_info=experiment_info,
        )

    def _send_training_finished_message(
        self, success: bool, task_id: int, experiment_info: dict
    ) -> TrainFinishedMessage:
        return TrainFinishedMessage(
            success=success, task_id=task_id, experiment_info=experiment_info or {}
        )

    # ------------------------------------------------------------------
    # Automation ---------------------------------------------------
    # ------------------------------------------------------------------
    def update_automation_details(self) -> Tuple[int, str, int, str]:
        enabled, period, interval, min_batch, sec = self.automation.get_details()
        if enabled is not None:
            self.show_automation_badge()
            self.update_property("Run every", f"{interval} {period}", highlight=True)
            if min_batch is not None:
                self.update_property("Min batch size", f"{min_batch}", highlight=True)
            else:
                self.remove_property_by_key("Min batch size")
        else:
            self.hide_automation_badge()
            self.remove_property_by_key("Run every")
            self.remove_property_by_key("Min batch size")

    def apply_automation(self) -> None:
        """
        Apply the automation function to the MoveLabeled node.
        """
        self.automation.apply(func=self.run)
        self.update_automation_details()

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, app_id: int, model_id: int, task_id: int) -> None:
        """Start the task to train data from the training project."""
        self._save_train_settings()

        # add task to tasks history
        task_info = self.api.task.get_info_by_id(task_id)

        train_collection = self.gui.widget.train_collections
        train_collection = train_collection[0] if train_collection else None
        val_collection = self.gui.widget.val_collections
        val_collection = val_collection[0] if val_collection else None

        images_count = "N/A"
        if train_collection and val_collection:
            train_imgs = self.api.entities_collection.get_items(
                train_collection, CollectionTypeFilter.DEFAULT
            )
            val_imgs = self.api.entities_collection.get_items(
                val_collection, CollectionTypeFilter.DEFAULT
            )
            images_count = f"train: {len(train_imgs)}, val: {len(val_imgs)}"
        classes_count = len(self._train_settings.get("classes", []))

        task = {
            "task_id": task_id,
            "task_info": task_info,
            "model_id": model_id,
            "status": "started",
            "agent_id": self._train_settings.get("agentId"),
            "classes_count": f"{classes_count} class{'s' if classes_count != 1 else ''}",
            "images_count": images_count,
        }
        self.history.add_task(task=task)

        self.update_badge_by_key(key="Training", label="Starting Application", badge_type="info")
        # Avoid using automation.apply()
        threading.Thread(target=self._monitor_training, args=(task_id, 5), daemon=True).start()


    # ------------------------------------------------------------------
    # Progress ---------------------------------------------------------
    # ------------------------------------------------------------------
    def _monitor_training(self, task_id: int, interval_sec: int = 10) -> None:
        """Use base poller; on finish, fetch experiment info and publish events."""
        success = self.poll_task_progress(task_id=task_id, interval_sec=interval_sec)
        if success:
            try:
                time.sleep(5)  # wait for experiment to be registered
                experiment_info = self.api.nn.get_experiment_info(task_id)
                experiment_info = experiment_info.to_json()
            except Exception as e:
                logger.warning(
                    f"Failed to get experiment info for task_id={task_id}: {repr(e)}"
                )
                experiment_info = None
            self._send_training_finished_message(
                success=True, task_id=task_id, experiment_info=experiment_info
            )
            self._send_training_output_message(
                success=True, task_id=task_id, experiment_info=experiment_info
            )

    def get_task_progress_label(self, task_id: int) -> Optional[str]:
        """Custom label mapping for training phases based on task progress widgets/name."""

        # Messages mapping
        start_messages = [
            "Application is started",
            "Application is started ..."
        ]
        prepare_messages = [
            "Preparing data for training...",
            "Downloading input data",
            "Retrieving data from cache",
            "Applying train/val splits to project",
            "Processing splits",
            "Downloading model files",
            "Training is in progress...",
        ]
        training_messages = ["Epochs", "Epoches", "Epoch"]
        finalize_messages = [
            "Uploading demo files to Team Files",
            "Uploading train artifacts to Team Files",
            "Finalizing and uploading training artifacts...",
            "Uploading training artifacts to Team Files",
        ]
        eval_messages = [
            "Starting Model Benchmark evaluation",
            "Evaluation: Downloading GT annotations",
            "Visualizations: Adding tags to predictions",
            "Speedtest: Running speedtest for batch sizes",
            "Uploading visualizations to visualizations",
            "Visualizations: Creating difference project",
            "Inferring model...",
        ]
        export_messages = [
            f"Converting to {RuntimeType.ONNXRUNTIME}",
            f"Converting to {RuntimeType.TENSORRT}",
        ]
        complete_messages = ["Training completed"]
        error_messages = ["Ready for training"] # message appears when training have failed

        task_progress = self._get_task_progress(task_id)
        message = task_progress.get("name")
        if not message:
            return None

        # Map message to label
        if message in start_messages:
            return "Starting Application"
        if message in prepare_messages:
            return "Preparing Data"
        if message in training_messages:
            current = task_progress.get("current")
            total = task_progress.get("total")
            if current is not None and total is not None:
                return f"{current} / {total} epochs"
        if message in finalize_messages or message.startswith("Uploading"):
            return "Finalizing"
        if message in eval_messages or message.startswith("Downloading annotations from"):
            return "Evaluating Model"
        if message in export_messages:
            return "Exporting Model"
        if message in complete_messages:
            return "Completed"
        if message in error_messages:
            return "Failed"
        return None           

    # ------------------------------------------------------------------
    # Utils ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _save_train_settings(self):
        """
        Extract training configuration from the embedded NewExperiment widget and store it
        inside the node so that it can be reused later
        """
        try:
            self._train_settings = self.gui.widget.get_train_settings()
            logger.info("Training settings saved.")
        except Exception as e:
            logger.warning(f"Failed to save training settings: {repr(e)}")


