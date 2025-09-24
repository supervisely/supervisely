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
        self.gui.project_table.set_project_filter(
            lambda p: p.id != self.project_id and p.items_count > 0
        )
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
    # def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
    #     """Returns a dictionary of methods that can be used for subscribing to events."""
    #     return {
    #         "add_training_data_id": self.start_task,
    #     }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "add_training_data_id": self.start_task,
        }

    def start_task(self) -> None:
        """Start the task to copy training data from one project to another."""
        if not self.settings_data:
            return

        team_id = sly_env.team_id()
        workspace_id = sly_env.workspace_id()
        task_id = sly_env.task_id()

        settings_data = self.settings_data
        dst_project_id = self.project_id

        src_project_id = settings_data["project_id"]
        src_workspace_id = settings_data["workspace_id"]
        src_team_id = settings_data["team_id"]

        self.show_in_progress_badge()
        train_split, val_split = self.gui.splits_widget.get_splits()
        if train_split is None and val_split is None:
            raise RuntimeError("No train/val split selected. Cannot start the task.")
        img_ids = [item_info.id for item_info in train_split + val_split]

        module_info = self.api.app.get_ecosystem_module_info(slug=self.APP_SLUG)
        params = {
            "state": {
                "action": "merge",
                "items": [{"id": img_id, "type": "image"} for img_id in img_ids],
                "source": {
                    "team": {"id": src_team_id},
                    "workspace": {"id": src_workspace_id},
                    "project": {"id": src_project_id},
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
                    "saveIdsToProjectCustomData": True,
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
            args=(
                task_id,
                dst_project_id,
                train_split,
                val_split,
            ),
            daemon=True,
        )
        thread.start()

    def wait_task_complete(
        self,
        task_id: int,
        dst_project_id: int,
        train_split: List,
        val_split: List,
    ) -> TrainingDataAddedMessage:
        """Wait until the task is complete."""
        task_info_json = self.api.task.get_info_by_id(task_id)
        if task_info_json is None:
            logger.error(f"Task with ID {task_id} not found.")
            return

        current_time = time.time()
        while (task_status := self.api.task.get_status(task_id)) != self.api.task.Status.FINISHED:
            if task_status in [
                self.api.task.Status.ERROR,
                self.api.task.Status.STOPPED,
                self.api.task.Status.TERMINATING,
            ]:
                logger.error(f"Task {task_id} failed with status: {task_status}")
                break
            logger.info("Waiting for the Data Commander task to finish... Status: %s", task_status)
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

        uploaded_ids = self._get_uploaded_ids(dst_project_id, task_id)
        assert uploaded_ids, RuntimeError("Failed to fetch uploaded IDs.")

        infos = self.api.image.get_info_by_id_batch(uploaded_ids)

        train_hashes = [item_info.hash for item_info in train_split]
        val_hashes = [item_info.hash for item_info in val_split]

        train_image_ids = [info.id for info in infos if info.hash in train_hashes]
        val_image_ids = [info.id for info in infos if info.hash in val_hashes]

        self._add_new_collection(dst_project_id, train_image_ids, val_image_ids)
        self.gui.project_table.enable()

        return TrainingDataAddedMessage(
            project_id=self.project_id,
            train_ids=train_image_ids,
            val_ids=val_image_ids,
        )

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

    def _add_new_collection(self, project_id: int, train_ids: List, val_ids: List) -> None:
        """Create new collections for training and validation splits."""
        from supervisely.api.entities_collection_api import CollectionTypeFilter

        train_col, train_col_idx = get_last_split_collection(self.api, project_id, "train_")
        val_col, val_col_idx = get_last_split_collection(self.api, project_id, "val_")

        train_last_collection_items = self.api.entities_collection.get_items(
            train_col.id, CollectionTypeFilter.DEFAULT, project_id
        )
        val_last_collection_items = self.api.entities_collection.get_items(
            val_col.id, CollectionTypeFilter.DEFAULT, project_id
        )
        train_last_ids = [item.id for item in train_last_collection_items]
        val_last_ids = [item.id for item in val_last_collection_items]

        new_train_collection = self.api.entities_collection.create(
            project_id, f"train_{train_col_idx + 1:04d}"
        )
        new_val_collection = self.api.entities_collection.create(
            project_id, f"val_{val_col_idx + 1:04d}"
        )

        self.api.entities_collection.add_items(new_train_collection.id, train_last_ids + train_ids)
        self.api.entities_collection.add_items(new_val_collection.id, val_last_ids + val_ids)
