import threading
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.app.widgets import Dialog
from supervisely.project.image_transfer_utils import move_structured_images
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.components.move_labeled.automation import MoveLabeledAuto
from supervisely.solution.components.move_labeled.gui import MoveLabeledGUI
from supervisely.solution.components.move_labeled.history import MoveLabeledTasksHistory
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    MoveLabeledDataFinishedMessage,
)


class MoveLabeledNode(BaseCardNode):
    APP_SLUG = "supervisely-ecosystem/data-commander"
    """
    This class is a placeholder for the MoveLabeled node.
    It is used to move labeled data from one location to another.
    """
    title = "Move Labeled Data"
    description = "Move labeled and accepted images to the Training Project."
    icon = "zmdi zmdi-dns"
    icon_color = "#1976D2"
    icon_bg_color = "#E3F2FD"

    def __init__(
        self,
        src_project_id: int,
        dst_project_id: int,
        *args,
        **kwargs,
    ):

        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.src_project_id = src_project_id
        self.dst_project_id = dst_project_id
        self._images_to_move = []

        # --- core blocks --------------------------------------------------------
        self.automation = MoveLabeledAuto(self.start_task)
        self.gui = MoveLabeledGUI()
        self.modal_content = self.gui.widget  # for BaseCardNode
        self.history = MoveLabeledTasksHistory()

        # --- modals -------------------------------------------------------------
        self.modals = [
            self.gui.modal,
            self.automation.modal,
            self.history.modal,
            self.history.logs_modal,
        ]

        # --- node init ----------------------------------------------------------
        title = kwargs.pop("title", self.title)
        description = kwargs.pop("description", self.description)
        icon = kwargs.pop("icon", self.icon)
        icon_color = kwargs.pop("icon_color", self.icon_color)
        icon_bg_color = kwargs.pop("icon_bg_color", self.icon_bg_color)
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

        # @self.click
        # def on_automate_click():
        #     self.gui.modal.show()

        @self.automation.apply_button.click
        def on_automate_click():
            self.automation.modal.hide()
            self.apply_automation()

        @self.gui.run_btn.click
        def on_run_click():
            self.gui.modal.hide()
            self.start_task()

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "queue_info_updated",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "move_labeled_data_finished",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "move_labeled_data_finished": self.wait_task_complete,
            # "queue_info_updated": self.send_images_count_message,
        }

    def _available_subscribe_methods(self):
        """Returns a dictionary of methods that can be used as callbacks for subscribed events."""
        return {
            "queue_info_updated": self.set_images_to_move,
        }

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
        self.send_images_count_message(len(self._images_to_move))

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
        self.automation.apply()
        self.update_automation_details()

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def start_task(self) -> None:
        """Start the task to move labeled data from one project to another."""
        src = self.src_project_id
        dst = self.dst_project_id
        images = self._images_to_move
        if not images:
            return

        min_batch_size = self.automation.automate_min_batch_input.get_value()
        use_min_batch = self.automation.automate_min_batch.is_checked()
        if use_min_batch and len(images) < min_batch_size:
            logger.warning(f"Not enough images to move. {min_batch_size} < {len(images)}")
            return

        self.show_in_progress_badge()
        logger.info(f"Moving {len(images)} images (project ID:{src} → ID:{dst}).")

        dst_project = self.api.project.get_info_by_id(dst)
        ds_count = dst_project.datasets_count or 0
        ds_name = f"batch_{ds_count + 1}"
        dst_dataset = self.api.dataset.create(dst, ds_name, change_name_if_conflict=True)

        module_info = self.api.app.get_ecosystem_module_info(slug=self.APP_SLUG)
        params = {
            "state": {
                # "items": [{"id": image_id, "type": "image"} for image_id in images],
                # "items": [ds ids] #  (parent dataset ids),
                "source": {
                    "team": {"id": sly_env.team_id()},
                    "project": {"id": src},
                    "workspace": {"id": sly_env.workspace_id()},
                },
                "destination": {
                    "team": {"id": sly_env.team_id()},
                    # "dataset": {"id": dst_dataset.id},
                    "project": {"id": dst},
                    "workspace": {"id": sly_env.workspace_id()},
                },
                "options": {
                    "preserveSrcDate": False,
                    "cloneAnnotations": True,
                    "conflictResolutionMode": "rename",
                    # "filter": [{"id": image_id, "type": "image"} for image_id in images], # by item ids
                },
                "action": "move",
            }
        }
        task_info_json = self.api.task.start(
            agent_id=self.gui.agent_selector.get_value(),
            workspace_id=sly_env.workspace_id(),
            description=f"Solutions: {sly_env.task_id()}",
            module_id=module_info.id,
            params=params,
        )
        task_info_json = self.api.task.get_info_by_id(task_info_json["id"])

        try:
            task_info_json = {
                "id": task_info_json["id"],
                "startedAt": task_info_json["startedAt"],
                "images_count": len(images),
                "status": task_info_json["status"],
            }
            self.history.add_task(task_info_json)
        except Exception as e:
            logger.error(f"Failed to add task to history: {repr(e)}")

        task_id = task_info_json["id"]
        thread = threading.Thread(
            target=self.wait_task_complete,
            args=(task_id, dst_dataset.id, images),
            daemon=True,
        )
        thread.start()
