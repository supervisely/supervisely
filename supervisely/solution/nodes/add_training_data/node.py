import os
import random
import threading
import time
from typing import Callable, Dict, List, Literal, Optional, Union

from supervisely import env as sly_env
from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import TrainingDataAddedMessage
from supervisely.solution.nodes.add_training_data.gui import AddTrainingDataGUI
from supervisely.solution.utils import get_last_split_collection


class AddTrainingDataNode(BaseCardNode):
    APP_SLUG = "supervisely-ecosystem/data-commander"
    """
    Node to add data to training project
    """

    TITLE = "Add Training Data"
    DESCRIPTION = (
        "Add new data to the training project and split it into training and validation sets."
    )
    ICON = "mdi mdi-folder-multiple-plus"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(self, project_id: int, *args, **kwargs):
        """
        Initialize the AddTrainingData node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        """

        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.project_id = project_id
        self._click_handled = True
        self.settings_data = None
        # --- core blocks --------------------------------------------------------
        self.gui = AddTrainingDataGUI(api=self.api)
        self.modal_content = self.gui.widget

        # --- modals -------------------------------------------------------------
        self.modals = [self.gui.modal]

        # --- init Node ----------------------------------------------------------
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

        @self.gui.on_settings_saved
        def on_save_split_settings_click(settings_data):
            self.settings_data = settings_data
            self.start_task()

        @self.click
        def show_modal():
            self.gui.modal.show()

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "add_training_data_id",
                "type": "target",
                "position": "right",
                "label": "Add Training Data",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "add_training_data_id": self.start_task,
        }

    def start_task(self) -> None:
        """Start the task to copy training daata from one project to another."""
        if not self.settings_data:
            return
        team_id = sly_env.team_id()
        workspace_id = sly_env.workspace_id()
        task_id = sly_env.task_id()

        settings_data = self.settings_data

        dst_project_id = self.project_id
        src_project_id = settings_data["project_id"]
        src_dataset_ids = settings_data["dataset_ids"]
        src_workspace_id = settings_data["workspace_id"]
        src_team_id = settings_data["team_id"]
        # replicate_structure = settings_data.get("replicate_structure", False)
        replicate_structure = False  # @TODO: add logic later
        train_split, val_split = settings_data.get("splits_ids", (None, None))
        if train_split is None and val_split is None:
            raise RuntimeError("No train/val split selected. Cannot start the task.")
        train_items, val_items = settings_data.get("splits_item", ([], []))

        self.show_in_progress_badge()

        module_info = self.api.app.get_ecosystem_module_info(slug=self.APP_SLUG)
        params = {
            "state": {
                "action": "copy" if not replicate_structure else "merge",
                # "items": [{"id": ds_id, "type": "dataset"} for ds_id in src_dataset_ids],
                "items": [{"id": img_id, "type": "item"} for img_id in train_split + val_split],
                "source": {
                    "team": {"id": src_team_id},
                    "workspace": {"id": src_workspace_id},
                    "project": {"id": src_project_id},
                    # "dataset": {"id": 16712, "parentIds": None},
                },
                "destination": {
                    "team": {"id": team_id},
                    "workspace": {"id": workspace_id},
                    "project": {"id": dst_project_id},
                },
                "options": {
                    "preserveSrcDate": False,
                    "cloneAnnotations": True,
                    "conflictResolutionMode": "rename",
                },
            }
        }
        agent = self.api.agent.get_list_available(team_id, True)[0]
        task_info_json = self.api.task.start(
            agent_id=agent.id,
            workspace_id=workspace_id,
            description=f"Solutions: {task_id} - Add Training Data",
            module_id=module_info.id,
            params=params,
        )

        task_id = task_info_json["id"]
        thread = threading.Thread(
            target=self.wait_task_complete,
            args=(task_id, dst_project_id, train_items, val_items),
            daemon=True,
        )
        thread.start()

    def wait_task_complete(
        self, task_id: int, dst_project_id: int, train_items: List, val_items: List
    ) -> TrainingDataAddedMessage:
        """Wait until the task is complete."""
        task_info_json = self.api.task.get_info_by_id(task_id)
        if task_info_json is None:
            logger.error(f"Task with ID {task_id} not found.")
            return

        # list project before task is started
        datasets = self.api.dataset.get_list(dst_project_id, recursive=True)
        dataset_images = {}
        for dataset_info in datasets:
            dataset_images[(dataset_info.id, dataset_info.name)] = self.api.image.get_list(
                dataset_info.id
            )

        current_time = time.time()
        while (task_status := self.api.task.get_status(task_id)) != self.api.task.Status.FINISHED:
            if task_status in [
                self.api.task.Status.ERROR,
                self.api.task.Status.STOPPED,
                self.api.task.Status.TERMINATING,
            ]:
                logger.error(f"Task {task_id} failed with status: {task_status}")
                break
            logger.info("Waiting for the Data Commander task to start... Status: %s", task_status)
            time.sleep(5)
            if time.time() - current_time > 300:  # 5 minutes timeout
                logger.warning(
                    "Timeout reached while waiting for the Data Commander task to start."
                )
                break

        self.hide_in_progress_badge()
        success = task_status == self.api.task.Status.FINISHED
        if not success:
            logger.error(f"Task {task_id} failed with status: {task_status}")

        # list project after task is finished
        new_datasets = self.api.dataset.get_list(dst_project_id, recursive=True)
        new_dataset_images = {}
        for dataset_info in new_datasets:
            new_dataset_images[(dataset_info.id, dataset_info.name)] = self.api.image.get_list(
                dataset_info.id
            )
        train_names = [item_info.name for item_info in train_items]
        val_names = [item_info.name for item_info in val_items]

        # find new images in the project and add them to the corresponding split collections
        train_image_ids, val_image_ids = [], []
        for (ds_id, ds_name), images in new_dataset_images.items():
            prev_images = dataset_images.get((ds_id, ds_name), [])
            if not prev_images:
                for image_info in images:
                    if image_info.name in train_names:
                        train_image_ids.append(image_info.id)
                    elif image_info.name in val_names:
                        val_image_ids.append(image_info.id)
                continue
            old_image_ids = {img_info.id for img_info in prev_images}
            current_images = {img_info.id for img_info in images}
            new_image_ids = list(current_images - old_image_ids)
            new_images = [img_info for img_info in images if img_info.id in new_image_ids]
            for image_info in new_images:
                if image_info.name in train_names:
                    train_image_ids.append(image_info.id)
                elif image_info.name in val_names:
                    val_image_ids.append(image_info.id)

        train_col, train_col_idx = get_last_split_collection(self.api, dst_project_id, "train")
        val_col, val_col_idx = get_last_split_collection(self.api, dst_project_id, "val")

        self.api.entities_collection.add_items(train_col.id, train_image_ids)
        self.api.entities_collection.add_items(val_col.id, val_image_ids)

        return TrainingDataAddedMessage(
            project_id=self.project_id,
            train_ids=train_image_ids,
            val_ids=val_image_ids,
        )
