import random
from dataclasses import dataclass
from typing import Dict, Optional

from supervisely.app.content import DataJson
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    RandomSplitsTable,
    SolutionCard,
)
from supervisely.solution.base_node import SolutionCardNode, SolutionElement


@dataclass
class SplitSettings:
    """Data class for storing random split settings."""

    train_percent: int = 80
    val_percent: int = 20

    def to_json(self) -> Dict:
        """Convert settings to dictionary for serialization."""
        return {
            "mode": "Random",
            "train_percent": self.train_percent,
            "val_percent": self.val_percent,
        }

    @classmethod
    def from_json(cls, data: Dict) -> "SplitSettings":
        """Create settings from dictionary."""
        return cls(
            train_percent=data.get("train_percent", 80),
            val_percent=data.get("val_percent", 20),
        )


class TrainValSplit(SolutionElement):
    """
    This class is a placeholder for the TrainValSplit node.
    It is used to move labeled data from one location to another.
    """

    def __init__(self, x: int, y: int, project_id: Optional[int] = None, *args, **kwargs):
        """
        Initialize the TrainValSplit node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        :param project_id: ID of the project from which labeled data will be split.
        """
        self.project_id = project_id
        self.main_widget = self._create_main_widget()
        self.split_settings = SplitSettings()
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        self.modals = [self.modal]
        super().__init__(*args, **kwargs)

        @self.ok_btn.click
        def save_split_settings():
            self.modal.hide()
            settings = self.get_split_settings()
            self.update_properties(settings)
            self.save_split_settings(settings)

        @self.card.click
        def show_split_modal():
            self.modal.show()

    @property
    def modal(self) -> Dialog:
        if not hasattr(self, "_modal"):
            self._modal = Dialog(
                title="Train/Val Split",
                content=self.main_widget,
            )
        return self._modal

    def _create_main_widget(self) -> Container:
        self.random_splits = RandomSplitsTable(0)
        ok_btn_container = Container([self.ok_btn], style="align-items: flex-end")
        return Container([self.random_splits, ok_btn_container])

    @property
    def ok_btn(self) -> Button:
        if not hasattr(self, "_ok_btn"):
            self._ok_btn = Button("Save", plain=True)
        return self._ok_btn

    @property
    def card(self) -> SolutionCard:
        if not hasattr(self, "_card"):
            self._card = self._create_card()
        return self._card

    def _create_card(self) -> SolutionCard:
        """Creates the SolutionCard for Train/Val Split."""
        return SolutionCard(
            title="Train/Val Split",
            tooltip=self._create_tooltip(),
            width=250,
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        """Creates the tooltip for the Train/Val Split card."""
        return SolutionCard.Tooltip(
            description="Split dataset into Train and Validation sets for model training. Datasets structure mirrors the Input Project with splits organized in corresponding Collections (e.g., 'train_1', 'val_1', etc.).",
        )

    def get_json_data(self) -> dict:
        return {
            "project_id": self.project_id,
            "split_settings": self.split_settings.to_json(),
        }

    def update_properties(self, settings: SplitSettings):
        """Update node properties with current split settings."""
        counts = self.random_splits.get_splits_counts()
        train = settings.train_percent
        val = settings.val_percent
        train_count = counts.get("train", 0)
        val_count = counts.get("val", 0)
        self.node.update_property("mode", f"Random ({train}% train, {val}% val)", highlight=True)
        self.node.update_property("train", f"{train_count} image{'' if train_count == 1 else 's'}")
        self.node.update_property("val", f"{val_count} image{'' if val_count == 1 else 's'}")

    def set_items_count(self, items_count: int):
        """Set the number of items in the random splits table."""
        self.random_splits.set_items_count(items_count)
        settings = self.get_split_settings()
        self.update_properties(settings)
        self.save_split_settings(settings)

    def get_split_settings(self) -> SplitSettings:
        """Get split settings from GUI."""
        train_percent = self.random_splits.get_train_split_percent()
        val_percent = self.random_splits.get_val_split_percent()
        return SplitSettings(train_percent=train_percent, val_percent=val_percent)

    def save_split_settings(self, settings: Optional[SplitSettings] = None):
        """Save split settings to node state."""
        if settings is None:
            settings = self.get_split_settings()
        self.split_settings = settings
        DataJson()[self.widget_id]["split_settings"] = settings.to_json()
        DataJson().send_changes()

    def split(self, items: list, random_selection: bool = True) -> Dict[str, list]:
        """
        Split the given items into train and validation sets.

        :param items: List of items to split.
        :return: Dictionary with train and validation items.
        """
        settings = self.get_split_settings()
        train_count = int(len(items) * settings.train_percent / 100)
        val_count = len(items) - train_count
        if random_selection:
            random.shuffle(items)
        train_items = items[:train_count]
        val_items = items[train_count : train_count + val_count]
        return {"train": train_items, "val": val_items}
