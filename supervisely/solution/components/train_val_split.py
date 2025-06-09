from dataclasses import dataclass
from typing import Dict, Optional

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


class TrainValSplitGUI:
    def __init__(self, items_count: int = 0):
        self.items_count = items_count

        self.modal = self._create_modal()
        self.card = self._create_card()

        @self.card.click
        def show_split_modal():
            self.modal.show()

    @property
    def ok_btn(self) -> Button:
        if not hasattr(self, "_ok_btn"):
            self._ok_btn = Button("Save", plain=True)
        return self._ok_btn

    def _create_modal(self) -> Dialog:
        """Creates the modal dialog for Train/Val Split."""
        self.random_splits = RandomSplitsTable(self.items_count)
        ok_btn_container = Container([self.ok_btn], style="align-items: flex-end")

        modal = Dialog(
            title="Train/Val Split",
            content=Container([self.random_splits, ok_btn_container]),
        )
        return modal

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


class TrainValSplit(SolutionElement):
    """
    This class is a placeholder for the TrainValSplit node.
    It is used to move labeled data from one location to another.
    """

    def __init__(self, x: int, y: int, project_id: Optional[int] = None):
        """
        Initialize the TrainValSplit node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        :param project_id: ID of the project from which labeled data will be split.
        """
        self.project_id = project_id
        self.gui = TrainValSplitGUI()
        self.split_settings = SplitSettings()
        self.node = SolutionCardNode(content=self.gui.card, x=x, y=y)
        self.modals = [self.gui.modal]
        super().__init__()

        @self.gui.ok_btn.click
        def save_split_settings():
            self.gui.modal.hide()
            settings = self.get_split_settings()
            self.update_properties(settings)
            self.save_split_settings(settings)

    def get_json_data(self) -> dict:
        """Return JSON data for the node."""
        return {
            "project_id": self.project_id,
            "split_settings": self.split_settings.to_json(),
        }

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
        settings = self.get_split_settings()
        self.update_properties(settings)
        self.save_split_settings(settings)

    def get_split_settings(self) -> SplitSettings:
        """Get split settings from GUI."""
        train_percent = self.gui.random_splits.get_train_split_percent()
        val_percent = self.gui.random_splits.get_val_split_percent()
        return SplitSettings(train_percent=train_percent, val_percent=val_percent)

    def save_split_settings(self, settings: Optional[SplitSettings] = None):
        """Save split settings to node state."""
        if settings is None:
            settings = self.get_split_settings()
        # self.state["split_settings"] = settings.to_json()
        # self.send_state_changes()
        self.split_settings = settings
