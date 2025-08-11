from typing import Callable, Dict, List, Optional, Tuple, Union

from supervisely.api.api import Api
from supervisely.sly_logger import logger
from supervisely.solution.base_node import SolutionElement, SolutionProjectNode
from supervisely.solution.components.project.automation import ProjectAutomation
from supervisely.solution.components.project.gui import ProjectGUI
from supervisely.solution.engine.events import on_event
from supervisely.solution.engine.models import (
    ImportFinishedMessage,
    MoveLabeledDataFinishedMessage,
    SampleFinishedMessage,
)


class ProjectNode(SolutionElement):
    """
    Project node for representing a Supervisely project in a solution.
    This node displays project information and provides links to view the project.
    """

    def __init__(
        self,
        project_id: int,
        title: str = "Input Project",
        description: str = "Centralizes all incoming data. Data in this project will not be modified.",
        is_training: bool = False,
        x: int = 0,
        y: int = 0,
        dataset_id: Optional[int] = None,
        refresh_interval: int = 30,
        tooltip_position: str = "left",
        *args,
        **kwargs,
    ):
        """
        Initialize the Project node.

        :param api: Supervisely API instance
        :param project_id: ID of the project to display
        :param title: Title for the project node
        :param description: Description of the project node
        :param is_training: Whether this is a training project (affects display)
        :param x: X coordinate of the node
        :param y: Y coordinate of the node
        :param dataset_id: Optional dataset ID to filter items in the project
        :param args: Additional positional arguments
        """
        self.api = Api.from_env()
        self.project_id = project_id

        # --- project info ------------------------------------------------------
        self.project = self.api.project.get_info_by_id(project_id)
        self.workspace_id = self.project.workspace_id
        self.dataset_id = dataset_id
        self.dataset = None
        if self.dataset_id is not None:
            self.dataset = self.api.dataset.get_info_by_id(self.dataset_id)

        self.title = title
        self.description = description
        self.is_training = is_training

        # --- core blocks --------------------------------------------------------
        self.automation = ProjectAutomation(project_id=self.project_id, func=self.update)
        self.gui = ProjectGUI(
            title=self.title,
            project=self.project,
            is_training=self.is_training,
            dataset=self.dataset,
            tooltip_position=tooltip_position,
        )
        self.node = SolutionProjectNode(content=self.gui.card, x=x, y=y)

        # --- modals -------------------------------------------------------------
        self.modals = []

        # --- refresh ------------------------------------------------------------
        self.refresh_interval = refresh_interval
        # self.apply_automation(sec=self.refresh_interval)

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Base Widget Methods ----------------------------------------------
    # ------------------------------------------------------------------
    def get_json_data(self) -> dict:
        """
        Returns the current data of the Project widget.
        """
        return {
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "is_training": self.is_training,
            "dataset_id": self.dataset_id,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Project widget.
        """
        return {}

    # ------------------------------------------------------------------
    # Update Methods ---------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "import_finished": self.update,
            "sample_finished": self.update,
            "train_val_split_finished": self.update,
        }

    def update(
        self,
        message: Union[
            ImportFinishedMessage,
            SampleFinishedMessage,
            MoveLabeledDataFinishedMessage,
        ],
    ) -> None:
        """
        Update the project node with new information.

        :param new_items_count: Optional count of newly added items
        """
        if not message.success or not message.items_count:
            logger.info("Skipping update project info due to unsuccessful import.")
            return
        new_items_count = message.items_count
        self.project = self.api.project.get_info_by_id(self.project_id)
        items_count = self.project.items_count or 0
        if isinstance(message, ImportFinishedMessage):
            preview_url = message.image_preview_url or self.project.image_preview_url
        else:
            preview_url = self.project.image_preview_url
        if self.dataset_id is not None:
            self.dataset = self.api.dataset.get_info_by_id(self.dataset_id)
            items_count = self.dataset.items_count or 0
            preview_url = self.dataset.image_preview_url

        if new_items_count is not None:
            self.node.update_property(key="Last update", value=f"+{new_items_count}")
            self.node.update_badge_by_key(key="Last update:", label=f"+{new_items_count}")
        self.node.update_property(key="Total", value=f"{items_count} images")

        # Update preview
        if self.is_training:
            train_items, val_items = self._get_train_val_items()
            self.gui.update_preview(
                [self._get_random_image_url(train_items), self._get_random_image_url(val_items)],
                [len(train_items), len(val_items)],
            )
        else:
            self.gui.update_preview([preview_url], [items_count or 0])

    def _get_train_val_collections(self) -> Tuple[List[int], List[int]]:
        """
        Returns the training and validation collections for the project.
        If the project is not a training project, returns empty lists.

        :return: Tuple of lists containing training and validation collection IDs.
        """
        train_collections = []
        val_collections = []

        if self.is_training:
            for collection_info in self.api.entities_collection.get_list(self.project_id):
                if collection_info.name.startswith("train_"):
                    train_collections.append(collection_info.id)
                elif collection_info.name.startswith("val_"):
                    val_collections.append(collection_info.id)

            if not train_collections and not val_collections:
                logger.warning("No training or validation collections found in the project.")

        return train_collections, val_collections

    def _get_train_val_items(self) -> Tuple[List, List]:
        """
        Returns the items in training and validation collections.

        :param dataset_id: Optional dataset ID to filter items
        :type dataset_id: Optional[int]
        :return: Tuple containing lists of training and validation items.
        """
        train_collections, val_collections = self._get_train_val_collections()
        train_items, val_items = [], []

        for collection_id in train_collections:
            images = self.api.entities_collection.get_items(collection_id, self.project_id)
            train_items.extend(images)

        for collection_id in val_collections:
            images = self.api.entities_collection.get_items(collection_id, self.project_id)
            val_items.extend(images)

        if self.dataset_id is not None:
            # Filter items by dataset ID if provided
            train_items = [item for item in train_items if item.dataset_id == self.dataset_id]
            val_items = [item for item in val_items if item.dataset_id == self.dataset_id]

        return train_items, val_items

    def _get_random_image_url(self, images: List) -> Optional[str]:
        """Get a random image URL from a list of images"""
        import random

        if not images:
            return None

        image = random.choice(images)
        if hasattr(image, "preview_url") and image.preview_url:
            return image.preview_url
        elif hasattr(image, "id"):
            # Try to get the preview URL directly from the API
            return self.api.image.get_preview_url(image.id, self.project_id)

        return None

    # ------------------------------------------------------------------
    # Automation --------------------------------------------------------
    # ------------------------------------------------------------------
    def apply_automation(self, sec: int, *args) -> None:
        """
        Apply the automation to refresh the project node periodically.

        :param sec: Interval in seconds for refreshing the project node
        """
        self.automation.apply(sec, *args)
