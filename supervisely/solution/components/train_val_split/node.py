import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.sly_logger import logger
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.train_val_split.gui import SplitSettings, TrainValSplitGUI
from supervisely.solution.engine.models import (
    LabelingQueueRefreshInfoMessage,
    MoveLabeledDataFinishedMessage,
    SampleFinishedMessage,
    TrainValSplitMessage,
)


class TrainValSplitNode(SolutionElement):
    """
    This class is a placeholder for the TrainValSplit node.
    It is used to move labeled data from one location to another.
    """

    def __init__(self, dst_project_id: int, x: int, y: int, *args, **kwargs):
        """
        Initialize the TrainValSplit node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        """
        self.api = Api.from_env()
        self.dst_project_id = dst_project_id
        self.split_settings = SplitSettings()
        self.gui = TrainValSplitGUI()
        self.card = self._build_card(
            title="Train/Val Split",
            tooltip_description="Split dataset into Train and Validation sets for model training. Datasets structure mirrors the Input Project with splits organized in corresponding Collections (e.g., 'train_1', 'val_1', etc.).",
        )
        self.node = SolutionCardNode(content=self.card, x=x, y=y)

        # --- modals -------------------------------------------------------------
        self.modals = [self.gui.modal]
        super().__init__(*args, **kwargs)

        @self.gui.ok_btn.click
        def save_split_settings():
            self.gui.modal.hide()
            settings = self.gui.get_split_settings()
            self.update_properties(settings)
            self.save_split_settings(settings)

        @self.card.click
        def show_split_modal():
            self.gui.modal.show()

    def get_json_data(self) -> dict:
        return {"split_settings": self.split_settings.to_json()}

    def update_properties(self, settings: SplitSettings):
        """Update node properties with current split settings."""
        counts = self.gui.random_splits.get_splits_counts()
        train = settings.train_percent
        val = settings.val_percent
        train_count = counts.get("train", 0)
        val_count = counts.get("val", 0)
        self.node.update_property("mode", f"Random ({train}% train, {val}% val)", highlight=True)
        self.node.update_property("train", f"{train_count} image{'' if train_count == 1 else 's'}")
        self.node.update_property("val", f"{val_count} image{'' if val_count == 1 else 's'}")

    def set_items_count(
        self, message: Union[SampleFinishedMessage, LabelingQueueRefreshInfoMessage]
    ) -> None:
        """Set the number of items in the random splits table."""
        if isinstance(message, SampleFinishedMessage):
            items_count = message.items_count
        elif isinstance(message, LabelingQueueRefreshInfoMessage):
            items_count = message.finished
        else:
            items_count = 0
        self.gui.random_splits.set_items_count(items_count)
        settings = self.gui.get_split_settings()
        self.update_properties(settings)
        self.save_split_settings(settings)

    def save_split_settings(self, settings: Optional[SplitSettings] = None):
        """Save split settings to node state."""
        if settings is None:
            settings = self.gui.get_split_settings()
        self.split_settings = settings
        DataJson()[self.widget_id]["split_settings"] = settings.to_json()
        DataJson().send_changes()

    def split(
        self,
        message: MoveLabeledDataFinishedMessage,
        random_selection: bool = True,
    ) -> MoveLabeledDataFinishedMessage:
        """
        Split the given items into train and validation sets.

        :param items: List of items to split.
        :return: Dictionary with train and validation items.
        """
        settings = self.gui.get_split_settings()
        if not message.items_count:
            logger.warning("No items to split. Returning empty splits.")
            # return TrainValSplitMessage(train=[], val=[]) # TODO:
        items = [img_id for img_ids in message.dst.values() for img_id in img_ids]
        train_count = int(len(items) * settings.train_percent / 100)
        val_count = len(items) - train_count
        if random_selection:
            random.shuffle(items)
        train_items = items[:train_count]
        val_items = items[train_count : train_count + val_count]
        self._add_to_collection(train_items, "train")
        self._add_to_collection(val_items, "val")

        return MoveLabeledDataFinishedMessage(
            success=True,
            src=message.src,
            dst=message.dst,
            items_count=len(train_items) + len(val_items),
        )

    def get_split_settings(self) -> SplitSettings:
        """Get split settings from GUI."""
        return self.gui.get_split_settings()

    def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "labeling_queue_info_refresh": [self.set_items_count],
            "move_labeled_data_finished": self.split,
        }

    def _available_publish_methods(self):
        """Returns a dictionary of methods that can be used for publishing events."""
        return {"train_val_split_finished": self.split}

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
