import threading
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.api.entities_collection_api import CollectionTypeFilter
from supervisely.api.task_api import TaskApi
from supervisely.app.widgets import Dialog, NewExperiment
from supervisely.project.image_transfer_utils import move_structured_images
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


class BaseTrainNode(BaseCardNode):
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

        # --- modals -------------------------------------------------------------
        self.modals = [
            self.gui.widget,
            self.automation.modal,
            self.history.modal,
            self.history.logs_modal,
        ]

        @self.click
        def on_click():
            self.gui.widget.visible = True
            # self.gui.modal.show()

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
        return {"data_versioning_project_id": self.get_data_versioning_project_id}

    def send_data_moving_finished_message(
        self,
        success: bool,
        items: List[int],
        items_count: int,
    ) -> MoveLabeledDataFinishedMessage:
        return MoveLabeledDataFinishedMessage(
            success=success,
            items=items,
            items_count=items_count,
        )

    # def send_images_count_message(self, count: int) -> LabelingQueueAcceptedImagesMessage:
    #     return LabelingQueueAcceptedImagesMessage(accepted_images=[i for i in range(count)])

    # publish event (may send Message object)
    def wait_task_complete(
        self,
        task_id: int,
        dataset_id: int,
        images: List[int],
    ) -> None:
        """Wait until the task is complete."""
        task_info_json = self.api.task.get_info_by_id(task_id)
        if task_info_json is None:
            logger.error(f"Task with ID {task_id} not found.")
            self.send_data_moving_finished_message(success=False, items=[], items_count=0)

        current_time = time.time()
        while (task_status := self.api.task.get_status(task_id)) != self.api.task.Status.FINISHED:
            if task_status in [
                self.api.task.Status.ERROR,
                self.api.task.Status.STOPPED,
                self.api.task.Status.TERMINATING,
            ]:
                logger.error(f"Task {task_id} failed with status: {task_status}")
                break
            logger.info("Waiting for the evaluation task to start... Status: %s", task_status)
            time.sleep(5)
            if time.time() - current_time > 30000:  # 500 minutes timeout
                logger.warning("Timeout reached while waiting for the evaluation task to start.")
                break

        try:
            task_info_json = {"id": task_id, "status": task_status.value}
            self.history.update_task(task_id=task_id, task=task_info_json)
            logger.info(f"Task {task_id} completed with status: {task_status.value}")
        except Exception as e:
            logger.error(f"Failed to update task history: {repr(e)}")

        self.hide_in_progress_badge()

        success = task_status == self.api.task.Status.FINISHED
        res = self.api.image.get_list(dataset_id=dataset_id)
        res = [img.id for img in res]
        if len(res) != len(images):
            logger.error(f"Not all images were moved. Expected {len(images)}, but got {len(res)}.")
            success = False

        if success:
            logger.info(f"Setting {len(res)} images as moved. Cleaning up the list.")
            self._images_to_move = []

        self.send_data_moving_finished_message(success=success, items=res, items_count=len(res))

    def get_data_versioning_project_id(self) -> Optional[int]:
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
            success=success, task_id=task_id, experiment_info=experiment_info
        )

    # subscribe event (may receive Message object)
    def set_images_to_move(self, message: LabelingQueueAcceptedImagesMessage) -> None:
        """
        Set the images to move based on the message.
        """
        if not message.accepted_images:
            logger.warning("No items to move. Returning empty list.")
            self._images_to_move = []
        else:
            self._images_to_move = message.accepted_images
        logger.info(f"Set {len(self._images_to_move)} images to move.")
        self.gui.set_items_count(len(self._images_to_move))
        self.update_property("Available items to move", f"{len(self._images_to_move)}")
        # self.send_images_count_message(len(self._images_to_move))

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

        task = {
            "id": task_id,
            "task_info": task_info,
            "model_id": self.gui.widget.model_id,
            "status": "started",
            "agent_id": self.gui.widget.agent_id,
            "classes_count": len(self.gui.widget.classes),
            "images_count": images_count,
        }
        self.history.add_task(task=task)

        # Avoid using automation.apply()
        threading.Thread(target=self._poll_train_progress, args=(task_id,), daemon=True).start()

    # ------------------------------------------------------------------
    # Utils ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _poll_train_progress(self, task_id: int, interval_sec: int = 10) -> None:
        """Poll task status every interval seconds until completion or failure."""
        # @ TODO: get train status from the task (fix send request on web progress status message)
        # train_status = self.api.task.send_request(task_id, "train_status", {})
        # print(f"Train status: {train_status}")

        while True:
            try:
                task_info = self.api.task.get_info_by_id(task_id)
            except Exception as e:
                logger.error(f"Failed to get task info for task_id={task_id}: {repr(e)}")
                break

            if task_info is None:
                logger.error(f"Task info is not found for task_id: {task_id}")
                break

            status = task_info.get("status")
            if status == TaskApi.Status.ERROR.value:
                self.update_badge_by_key(key="Status", label="Failed", badge_type="error")
                break
            if status in [TaskApi.Status.STOPPED.value, TaskApi.Status.TERMINATING.value]:
                self.update_badge_by_key(key="Status", label="Stopped", badge_type="warning")
                break
            if status == TaskApi.Status.CONSUMED.value:
                self.update_badge_by_key(key="Status", label="Consumed", badge_type="warning")
            elif status == TaskApi.Status.QUEUED.value:
                self.update_badge_by_key(key="Status", label="Queued", badge_type="warning")
            elif status == TaskApi.Status.FINISHED.value:
                self.update_badge_by_key(key="Status", label="Finished", badge_type="success")
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
                break
            else:
                self.update_badge_by_key(key="Status", label="Training...", badge_type="info")

            time.sleep(interval_sec)

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
