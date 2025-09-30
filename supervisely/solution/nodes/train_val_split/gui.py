from dataclasses import dataclass
from typing import Dict

from supervisely.app.content import DataJson
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    Field,
    RandomSplitsTable,
    Text,
    Widget,
)
from supervisely.solution.engine.modal_registry import ModalRegistry


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


class TrainValSplitGUI(Widget):
    """
    GUI for Train/Val Split node.
    Allows users to configure random split settings.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the TrainValSplit node.
        """
        self.split_settings = SplitSettings()
        self.content = self._create_main_widget()
        super().__init__(*args, **kwargs)
        self.set_items_count(0)  # Initialize with 0 items

        ModalRegistry().attach_settings_widget(owner_id=self.widget_id, widget=self.content)

    @property
    def modal(self) -> Dialog:
        return ModalRegistry().settings_dialog_small

    def open_modal(self):
        ModalRegistry().open_settings(owner_id=self.widget_id)

    def _create_main_widget(self) -> Container:
        info_text = Text(
            """
            This node automatically splits your dataset into Training and Validation sets based on the specified percentages. The
            dataset structure mirrors the Labeling Project, with splits organized in corresponding Collections (e.g., 'train_1', 'val_1', etc.).<br>
            <br>
            <strong>How it works:</strong><br>
            • The node processes most recently moved images from the Labeling project to the Training project and splits them into Training and Validation sets based on the specified percentages.<br>
            • The splits are saved in Collections named 'train_1', 'val_1', etc., within the Training project.<br>
            • The splits are also saved in the main train/val collections ('all_train', 'all_val') for further use in the pipeline.<br>
            <br>
        """
        )

        info_field = Field(
            Container([info_text], gap=15),
            title="How it works",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-help",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        self.random_splits = RandomSplitsTable(0)
        empty_text = Text("No available images for split...", color="#9E9E9E", status="warning")
        empty_text.hide()
        self.hide_splits_empty_text = lambda: empty_text.hide()
        self.show_splits_empty_text = lambda: empty_text.show()

        splits_field = Field(
            Container([self.random_splits, empty_text], gap=10),
            title="Random Split Settings",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-swap",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        ok_btn_container = Container([self.ok_btn], style="align-items: flex-end")
        return Container([info_field, splits_field, ok_btn_container], gap=20)

    @property
    def ok_btn(self) -> Button:
        if not hasattr(self, "_ok_btn"):
            self._ok_btn = Button("Save", plain=True)
        return self._ok_btn

    def get_json_data(self) -> dict:
        return {"split_settings": self.split_settings.to_json()}

    def get_json_state(self) -> dict:
        return {}

    def get_split_settings(self) -> SplitSettings:
        """Get split settings from GUI."""
        train_percent = self.random_splits.get_train_split_percent()
        val_percent = self.random_splits.get_val_split_percent()
        return SplitSettings(train_percent=train_percent, val_percent=val_percent)

    def save_split_settings(self, settings: SplitSettings):
        """Save split settings to node state."""
        self.split_settings = settings
        self.random_splits.set_train_split_percent(settings.train_percent)
        DataJson()[self.widget_id]["split_settings"] = settings.to_json()
        DataJson().send_changes()

    def set_items_count(self, count: int):
        """Set the count of available items for splitting."""
        self.random_splits.set_items_count(count)
        if count == 0:
            self.show_splits_empty_text()
        else:
            self.hide_splits_empty_text()
