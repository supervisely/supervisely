import random
from dataclasses import dataclass
from typing import Dict, Optional

from supervisely.app.content import DataJson
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.train_val_split.gui import (
    SplitSettings,
    TrainValSplitGUI,
)


class TrainValSplitNode(SolutionElement):
    """
    This class is a placeholder for the TrainValSplit node.
    It is used to move labeled data from one location to another.
    """

    def __init__(self, x: int, y: int, *args, **kwargs):
        """
        Initialize the TrainValSplit node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        """
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

    def set_items_count(self, items_count: int):
        """Set the number of items in the random splits table."""
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

    def split(self, items: list, random_selection: bool = True) -> Dict[str, list]:
        """
        Split the given items into train and validation sets.

        :param items: List of items to split.
        :return: Dictionary with train and validation items.
        """
        settings = self.gui.get_split_settings()
        train_count = int(len(items) * settings.train_percent / 100)
        val_count = len(items) - train_count
        if random_selection:
            random.shuffle(items)
        train_items = items[:train_count]
        val_items = items[train_count : train_count + val_count]
        return {"train": train_items, "val": val_items}

    def get_split_settings(self) -> SplitSettings:
        """Get split settings from GUI."""
        return self.gui.get_split_settings()
