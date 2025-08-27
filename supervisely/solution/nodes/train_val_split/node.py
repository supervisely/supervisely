import random
from typing import Callable, Dict, List, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    MoveLabeledDataFinishedMessage,
    SampleFinishedMessage,
)
from supervisely.solution.nodes.train_val_split.gui import (
    SplitSettings,
    TrainValSplitGUI,
)


class TrainValSplitNode(BaseCardNode):
    """
    This class is a placeholder for the TrainValSplit node.
    It is used to move labeled data from one location to another.
    """

    title = "Train/Val Split"
    description = "Split dataset into Train and Validation sets for model training. Datasets structure mirrors the Input Project with splits organized in corresponding Collections (e.g., 'train_1', 'val_1', etc.)."
    icon = "mdi mdi-set-split"
    icon_color = "#1976D2"
    icon_bg_color = "#E3F2FD"

    def __init__(self, dst_project_id: int, *args, **kwargs):
        """
        Initialize the TrainValSplit node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        """

        # --- core blocks --------------------------------------------------------
        self.gui = TrainValSplitGUI()
        self.modal_content = self.gui.widget

        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.dst_project_id = dst_project_id
        self._click_handled = True

        # --- modals -------------------------------------------------------------
        self.modals = [self.gui.modal]

        # --- init Node ----------------------------------------------------------
        title = kwargs.pop("title", "Train/Val Split")
        description = kwargs.pop(
            "description",
            "Split dataset into Train and Validation sets for model training. Datasets structure mirrors the Input Project with splits organized in corresponding Collections (e.g., 'train_1', 'val_1', etc.).",
        )
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

        @self.gui.ok_btn.click
        def on_save_split_settings_click():
            self.gui.modal.hide()
            self.save_split_settings()

        @self.click
        def show_split_modal():
            self.gui.modal.show()

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "train_val_split_items_count",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
            {
                "id": "train_val_split_finished",
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
            "move_labeled_data_finished": self.split,
            "train_val_split_items_count": self.set_items_count,
        }

    def _available_publish_methods(self):
        """Returns a dictionary of methods that can be used for publishing events."""
        return {"train_val_split_finished": self.send_train_val_split_finished_message}

    def split(
        self,
        message: MoveLabeledDataFinishedMessage,
        random_selection: bool = True,
    ) -> MoveLabeledDataFinishedMessage:
        """
        Split the given items into train and validation sets.

        :param message: Message containing the items to split.
        :type message: MoveLabeledDataFinishedMessage
        :param random_selection: Whether to randomly select items for the split.
        :type random_selection: bool
        :raises ValueError: If no items are provided in the message.
        :return: Returns the MoveLabeledDataFinishedMessage object
        :rtype: MoveLabeledDataFinishedMessage
        """
        if not message.success:
            logger.error("Failed to move labeled data. Cannot perform split.")
            return MoveLabeledDataFinishedMessage(success=False, items=[], items_count=0)
        settings = self.gui.get_split_settings()
        if not message.items_count:
            logger.warning("No items to split. Returning empty splits.")
            # return TrainValSplitMessage(train=[], val=[]) # TODO:
        items = message.items
        train_count = int(len(items) * settings.train_percent / 100)
        val_count = len(items) - train_count
        if random_selection:
            random.shuffle(items)
        train_items = items[:train_count]
        val_items = items[train_count : train_count + val_count]
        self._add_to_collection(train_items, "train")
        self._add_to_collection(val_items, "val")
        logger.info(
            f"Split {len(items)} items into {len(train_items)} train and {len(val_items)} val items."
        )
        self.set_items_count()

        return MoveLabeledDataFinishedMessage(
            success=True,
            items=train_items + val_items,
            items_count=len(train_items) + len(val_items),
        )

    def set_items_count(
        self, message: Union[SampleFinishedMessage, LabelingQueueAcceptedImagesMessage] = None
    ) -> None:
        """Set the number of items in the random splits table."""
        if isinstance(message, SampleFinishedMessage):
            items_count = message.items_count
        elif isinstance(message, LabelingQueueAcceptedImagesMessage):
            items_count = len(message.accepted_images)
        else:
            items_count = 0
        self.gui.set_items_count(items_count)
        self.save_split_settings()

    def send_train_val_split_finished_message(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _add_to_collection(
        self,
        image_ids: List[int],
        split_name: Literal["train", "val"],
    ) -> None:
        """
        Add the MoveLabeled node to a collection.
        """
        if not image_ids:
            logger.warning("No images to add to collection.")
            return
        collections = self.api.entities_collection.get_list(self.dst_project_id)

        main_collection_name = f"all_{split_name}"
        main_collection = None

        last_batch_index = 0
        for collection in collections:
            if collection.name == main_collection_name:
                main_collection = collection
            elif collection.name.startswith(f"{split_name}_"):
                last_batch_index = max(last_batch_index, int(collection.name.split("_")[-1]))

        if main_collection is None:
            main_collection = self.api.entities_collection.create(
                self.dst_project_id, main_collection_name
            )
            logger.info(f"Created new collection '{main_collection_name}'")

        self.api.entities_collection.add_items(main_collection.id, image_ids)

        batch_collection_name = f"{split_name}_{last_batch_index + 1}"
        batch_collection = self.api.entities_collection.create(
            self.dst_project_id, batch_collection_name
        )
        logger.info(f"Created new collection '{batch_collection_name}'")

        self.api.entities_collection.add_items(batch_collection.id, image_ids)

        logger.info(f"Added {len(image_ids)} images to {split_name} collections")
        self.api.entities_collection.add_items(batch_collection.id, image_ids)

        logger.info(f"Added {len(image_ids)} images to {split_name} collections")

    def _update_properties(self, settings: SplitSettings):
        """Update node properties with current split settings."""
        counts = self.gui.random_splits.get_splits_counts()
        train = settings.train_percent
        val = settings.val_percent
        train_count = counts.get("train", 0)
        val_count = counts.get("val", 0)
        self.update_property("mode", f"Random ({train}% train, {val}% val)", highlight=True)
        self.update_property("train", f"{train_count} image{'' if train_count == 1 else 's'}")
        self.update_property("val", f"{val_count} image{'' if val_count == 1 else 's'}")

    def save_split_settings(self, settings: Optional[SplitSettings] = None):
        """Save split settings to node state."""
        if settings is None:
            settings = self.gui.get_split_settings()
        self.gui.save_split_settings(settings)
        self._update_properties(settings)
