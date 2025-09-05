import os
import random
from typing import Callable, Dict, List, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    TrainingDataAddedMessage,
)
from supervisely.solution.nodes.add_training_data.gui import (
    AddTrainingDataGUI,
)
from supervisely import env as sly_env
import threading
import time
from supervisely.app.widgets.train_val_splits.train_val_splits import ItemInfo
from supervisely import Project, Dataset

class AddTrainingDataNode(BaseCardNode):
    APP_SLUG = "supervisely-ecosystem/data-commander"
    """
    Node to add data to training project
    """

    TITLE = "Add Training Data"
    DESCRIPTION = "Add new data to the training project and split it into training and validation sets."
    ICON = "zmdi zmdi-collection-folder-image"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(self, dst_project_id: int, *args, **kwargs):
        """
        Initialize the AddTrainingData node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        """


        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.dst_project_id = dst_project_id
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
                "id": "training_data_input",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
            {
                "id": "training_data_finished",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "training_data_added": self.start_task,
        }

    def _available_publish_methods(self):
        """Returns a dictionary of methods that can be used for publishing events."""
        return {"train_val_split_finished": self.send_data_copied_message}
    
    def send_data_copied_message(self):
        pass

    def start_task(self) -> None:
        """Start the task to copy training daata from one project to another."""
        if not self.settings_data:
            return
        team_id = sly_env.team_id()
        workspace_id = sly_env.workspace_id()
        task_id = sly_env.task_id()

        settings_data = self.settings_data

        src = settings_data['project_id']
        dst = self.dst_project_id

        train_set, val_set = settings_data['splits']
        train_ids = self._get_ids_by_iteminfos(src ,train_set)
        val_ids = self._get_ids_by_iteminfos(src, val_set)

        self.show_in_progress_badge()

        ds_name = "Training Data"
        dst_dataset = self.api.dataset.create(dst, ds_name, change_name_if_conflict=True)

        module_info = self.api.app.get_ecosystem_module_info(slug=self.APP_SLUG)
        params = {
            "state": {
                "action": "copy",
                "items": [{"id": image_id, "type": "image"} for image_id in train_ids + val_ids],
                "source": {
                    "team": {"id": team_id},
                    "project": {"id": src},
                    "workspace": {"id": workspace_id},
                },
                "destination": {
                    "team": {"id": team_id},
                    "project": {"id": dst},
                    "workspace": {"id": workspace_id},
                    # "dataset": {"id": dst_dataset.id},
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
            args=(task_id, dst_dataset.id, train_set, val_set),
            daemon=True,
        )
        thread.start()

    def wait_task_complete(
        self,
        task_id: int,
        dataset_id: int,
        train_set: List[ItemInfo],
        val_set: List[ItemInfo],
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
            logger.info("Waiting for the Data Commander task to start... Status: %s", task_status)
            time.sleep(5)
            if time.time() - current_time > 300:  # 5 minutes timeout
                logger.warning("Timeout reached while waiting for the Data Commander task to start.")
                break

        self.hide_in_progress_badge()

        success = task_status == self.api.task.Status.FINISHED
        res = self.api.image.get_list(dataset_id=dataset_id)
        res_item_name_to_id = {item.name: item.id for item in res}
        train_ids = [res_item_name_to_id[item.name] for item in train_set if item.name in res_item_name_to_id]
        val_ids = [res_item_name_to_id[item.name] for item in val_set if item.name in res_item_name_to_id]
        items_cnt = len(train_set) + len(val_set)
        res = [img.id for img in res]
        if len(res) != items_cnt:
            logger.error(f"Not all images were moved. Expected {items_cnt}, but got {len(res)}.")
            success = False

        if success:
            logger.info(f"Setting {len(res)} images as moved. Cleaning up the list.")
            self._images_to_move = []

        return TrainingDataAddedMessage(
            project_id=self.dst_project_id,
            dataset_ids=[dataset_id],
            splits={
                "train": train_ids,
                "val": val_ids,
            }
        )

    def _get_ids_by_iteminfos(self, project_id, items: List[ItemInfo]) -> List[int]:
        from collections import defaultdict
        datasets_to_list = list(set([item.dataset_name for item in items]))
        filters = [{"field": "name", "operator": "in", "value": datasets_to_list}]
        datasets = self.api.dataset.get_list(project_id, filters=filters, recursive=True)
        dataset_name_to_id = {ds.name: ds.id for ds in datasets}
        dataset_id_to_item_names = defaultdict(list)
        for item in items:
            dataset_info = dataset_name_to_id.get(item.dataset_name)
            if dataset_info is None:
                raise ValueError(f"Dataset {item.dataset_name} not found in project {project_id}")
            dataset_id_to_item_names[dataset_info].append(item.name)

        ds_id_to_image_ids = {}
        for dataset_id, item_names in dataset_id_to_item_names.items():
            filters = [{"field": "name", "operator": "in", "value": item_names}]
            images = self.api.image.get_list(dataset_id, filters=filters)
            ds_id_to_image_ids[dataset_id] = [img.id for img in images]

        return [img_id for ids in ds_id_to_image_ids.values() for img_id in ids]