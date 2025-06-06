import random
from typing import List, Optional, Tuple
from venv import logger

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.app.widgets import Button
from supervisely.solution.base_node import SolutionProjectNode


class ProjectNode(SolutionProjectNode):
    """
    ProjectNode node for importing data using simple D&D widget.
    This node also allows users to import data from a specified path in cloud storage, agent or team files,
    and manage import tasks.
    """

    # APP_SLUG = "supervisely-ecosystem/main-import"
    # JOB_ID = "manual_import_job"

    def __init__(
        self,
        x: int,
        y: int,
        project_id: int,
        title="Input Project",
        description="Centralizes all incoming data. Data in this project will not be modified.",
        is_training_project=False,
    ):
        """
        Initialize the ProjectNode node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        :param project_id: ID of the project to import data into.
        :param widget_id: Optional widget ID for the node.
        """
        self.api = Api.from_env()
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)
        self.workspace_id = self.project.workspace_id
        self.title = title
        self.description = description
        self.is_training = is_training_project

        super().__init__(x, y)

    def _create_gui(self):
        """
        Initialize the widgets for the Manual Import node.
        """
        stats_url = self.project.url.replace("datasets", "stats/datasets")
        tooltip_widgets = [
            Button(
                "Open project",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=self.project.url,
                button_type="text",
            ),
            Button(
                "QA stats",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=stats_url,
                button_type="text",
            ),
        ]
        tooltip = self.card_cls.Tooltip(
            description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time..",
            content=tooltip_widgets,
        )

        if self.is_training:
            items_count, preview_url = [0, 0], [None, None]

        if self.is_training:
            train, val = self._get_train_val_items()
            train_img = None if not train else random.choice(list(train))
            val_img = None if not val else random.choice(list(val))
            items_count = [len(set(train)), len(set(val))]
            # self.update_preview([train_img, val_img], items_count)

        else:
            items_count = [self.project.items_count or 0]
            preview_url = [self.project.image_preview_url]

        self.card = self.card_cls(
            title="Manual Drag & Drop Import",
            preview_url=preview_url,
            items_count=items_count,
            project_id=self.project.id,
            width=270 if self.is_training else 180,
            tooltip=tooltip,
            tooltip_position="left",
        )

    def get_json_data(self) -> dict:
        """
        Returns the current data of the Manual Import widget.
        """
        return {
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Manual Import widget.
        """
        return {}

    def _get_train_val_collections(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        Returns the training and validation collections for the project.
        :return: Tuple of lists containing training and validation collection IDs.
        """
        if self.is_training:
            train_collections = []
            val_collections = []
            for collection_info in self.api.entities_collection.get_list(self.project_id):
                if collection_info.name.startswith("train_"):
                    train_collections.append(collection_info.id)
                elif collection_info.name.startswith("val_"):
                    val_collections.append(collection_info.id)
            if not train_collections and not val_collections:
                logger.warning("No training or validation collections found in the project.")
            return train_collections, val_collections
        else:
            return None, None

    def _get_train_val_items(self) -> Tuple[List, List]:
        """
        Returns the count of items in training and validation collections.
        :return: Tuple containing counts of training and validation items.
        """
        train_collections, val_collections = self._get_train_val_collections()
        if not train_collections and not val_collections:
            logger.warning("No training or validation collections found in the project.")
            return [], []
        train, val = [], []
        for collection_id in train_collections:
            images = self.api.entities_collection.get_items(collection_id, self.project_id)
            train.extend([img.id for img in images])
        for collection_id in val_collections:
            images = self.api.entities_collection.get_items(collection_id, self.project_id)
            val.extend([img.id for img in images])
        return train, val
