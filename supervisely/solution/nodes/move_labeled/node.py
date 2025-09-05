import random
import threading
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import supervisely.io.env as sly_env
from supervisely.api.api import Api
from supervisely.app.widgets import Dialog
from supervisely.project.image_transfer_utils import move_structured_images
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    MoveLabeledDataFinishedMessage,
)
from supervisely.solution.nodes.move_labeled.automation import MoveLabeledAuto
from supervisely.solution.nodes.move_labeled.gui import MoveLabeledGUI
from supervisely.solution.nodes.move_labeled.history import MoveLabeledTasksHistory


class MoveLabeledNode(BaseCardNode):
    APP_SLUG = "supervisely-ecosystem/data-commander"
    """
    This class is a placeholder for the MoveLabeled node.
    It is used to move labeled data from one location to another.
    """
    TITLE = "Move Labeled Data"
    DESCRIPTION = "Move labeled and accepted images to the Training Project."
    ICON = "mdi mdi-folder-move"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

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
        self._train_percent = None
        self._val_percent = None
        self._click_handled = True

        # --- core blocks --------------------------------------------------------
        self.automation = MoveLabeledAuto()
        self.gui = MoveLabeledGUI()
        self.modal_content = self.gui.widget  # for BaseCardNode
        self.history = MoveLabeledTasksHistory()

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

        @self.click
        def on_click():
            self.gui.modal.show()

        @self.gui.run_btn.click
        def on_run_click():
            self.gui.modal.hide()
            self.run()

        # --- modals -------------------------------------------------------------
        self.modals = [
            self.gui.modal,
            self.automation.modal,
            self.history.modal,
            self.history.logs_modal,
        ]

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
                "id": "accepted_images",
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
            "move_labeled_data_finished": self.send_data_moving_finished_message,
            # "queue_info_updated": self.send_images_count_message,
        }

    def _available_subscribe_methods(self):
        """Returns a dictionary of methods that can be used as callbacks for subscribed events."""
        return {
            "accepted_images": self._update_images_to_move_info,
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

    # subscribe event (may receive Message object)
    def _update_images_to_move_info(self, message: LabelingQueueAcceptedImagesMessage) -> None:
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

        if message.train_split is not None:
            self._train_percent = message.train_split
        if message.val_split is not None:
            self._val_percent = message.val_split
        if self._train_percent is not None and self._val_percent is not None:
            logger.info(f"Train/Val split settings: {self._train_percent}%/{self._val_percent}%")

    # publish event (may send Message object)
    def wait_task_complete(self, task_id: int, src_images: List[int]) -> MoveLabeledDataFinishedMessage:
        """Wait until the task is complete."""
        task_info_json = self.api.task.get_info_by_id(task_id)
        if task_info_json is None:
            logger.error(f"Task with ID {task_id} not found.")
            return self.send_data_moving_finished_message(success=False, items=[], items_count=0)

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

        success = task_status == self.api.task.Status.FINISHED
        items = self._get_uploaded_ids(self.dst_project_id, task_id)

        try:
            task_info_json = {"id": task_id, "status": task_status.value}
            self.history.update_task(task_id=task_id, task=task_info_json)
            logger.info(f"Task {task_id} completed with status: {task_status.value}")
        except Exception as e:
            logger.error(f"Failed to update task history: {repr(e)}")

        try:
            if self._train_percent is not None and self._val_percent is not None:
                logger.info("Performing train/val split after moving data.")
                self.split(items)
        except Exception as e:
            logger.error(f"Failed to perform train/val split: {repr(e)}")

        self.hide_in_progress_badge()

        moved = len(items) if success else 0
        if success:
            logger.info(f"{moved} items moved successfully. Clearing the move list.")
            self._images_to_move = []

        if moved > 0:
            self.send_data_moving_finished_message(success=success, items=items, items_count=moved)
            self.api.image.remove_batch(src_images, batch_size=200)

    # ------------------------------------------------------------------
    # Automation ---------------------------------------------------
    # ------------------------------------------------------------------
    def _update_automation_details(self) -> None:
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
        self._update_automation_details()

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self) -> None:
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

        module_info = self.api.app.get_ecosystem_module_info(slug=self.APP_SLUG)
        params = {
            "state": {
                "items": [{"id": image_id, "type": "image"} for image_id in images],
                "source": {
                    "team": {"id": sly_env.team_id()},
                    "project": {"id": src},
                    "workspace": {"id": sly_env.workspace_id()},
                },
                "destination": {
                    "team": {"id": sly_env.team_id()},
                    "project": {"id": dst},
                    "workspace": {"id": sly_env.workspace_id()},
                },
                "options": {
                    "preserveSrcDate": False,
                    "cloneAnnotations": True,
                    "conflictResolutionMode": "skip",
                    "saveIdsToProjectCustomData": True,
                },
                "action": "merge",
            }
        }
        task_info_json = self.api.task.start(
            agent_id=self.gui.agent_selector.get_value(),
            workspace_id=sly_env.workspace_id(),
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
            args=(task_id, images),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Helpers ----------------------------------------------------------
    # ------------------------------------------------------------------
    def split(self, items: List[int], random_selection: bool = True):
        """Split the given items into train and validation sets."""
        if not items:
            logger.warning("No items to split.")
            return

        train_count = int(len(items) * self._train_percent / 100)
        val_count = len(items) - train_count
        if random_selection:
            random.shuffle(items)
        train = items[:train_count]
        val = items[train_count : train_count + val_count]
        self._add_to_collection(train, "all_train")
        self._add_to_collection(val, "all_val")
        self._add_to_collection(items, "batch")
        logger.info(f"Split {len(items)} items into {len(train)} train and {len(val)} val items.")

    def _add_to_collection(
        self,
        image_ids: List[int],
        split_name: Literal["all_train", "all_val", "batch"],
    ) -> None:
        """Add the MoveLabeled node to a collection."""
        if not image_ids:
            return
        collections = self.api.entities_collection.get_list(self.dst_project_id)

        main_col = None

        last_batch_idx = 0
        for collection in collections:
            if collection.name == split_name and split_name in ["all_train", "all_val"]:
                main_col = collection
            elif split_name == "batch":
                if collection.name.startswith("batch_"):
                    last_batch_idx = max(last_batch_idx, int(collection.name.split("_")[-1]))

        if main_col is None and split_name in ["all_train", "all_val"]:
            main_col = self.api.entities_collection.create(self.dst_project_id, split_name)
            logger.info(f"Created new collection '{split_name}'")
        elif split_name == "batch":
            batch_name = f"batch_{last_batch_idx + 1}"
            main_col = self.api.entities_collection.create(self.dst_project_id, batch_name)
            logger.info(f"Created new collection '{batch_name}'")

        self.api.entities_collection.add_items(main_col.id, image_ids)


    def _get_uploaded_ids(self, project_id: int, task_id: int) -> List[int]:
        """Get the IDs of images uploaded from the project's custom data."""
        project = self.api.project.get_info_by_id(project_id)
        if project is None:
            logger.warning(f"Project with ID {project_id} not found.")
            return []
        custom_data = project.custom_data or {}
        history = custom_data.get("import_history", {}).get("tasks", [])
        for record in history:
            if record.get("task_id") == task_id:
                break
        else:
            logger.warning(f"No import history found for task ID {task_id}.")
            return []
        uploaded_ids = []
        for ds in record.get("datasets", []):
            uploaded_ids.extend(list(map(int, ds.get("uploaded_images", []))))
        return uploaded_ids
