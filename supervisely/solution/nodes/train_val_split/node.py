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

    TITLE = "Train/Val Split"
    DESCRIPTION = "Split dataset into Train and Validation sets for model training. Datasets structure mirrors the Input Project with splits organized in corresponding Collections (e.g., 'train_1', 'val_1', etc.)."
    ICON = "mdi mdi-set-split"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

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
        self._accepted_images = []

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

        @self.gui.ok_btn.click
        def on_save_split_settings_click():
            self.gui.modal.hide()
            self.save_split_settings()
            self.send_accepted_images_message(
                accepted_images=self._accepted_images,
                splits=self.gui.get_split_settings(),
            )

        @self.click
        def show_split_modal():
            self.gui.modal.show()

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "accepted_images",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "accepted_images",
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
            # "move_labeled_data_finished": self.split,
            "accepted_images": self.set_items_count,
        }

    def _available_publish_methods(self):
        """Returns a dictionary of methods that can be used for publishing events."""
        return {"accepted_images": self.send_accepted_images_message}

    def send_accepted_images_message(
        self,
        accepted_images: List[int],
        splits: SplitSettings,
    ) -> LabelingQueueAcceptedImagesMessage:

        return LabelingQueueAcceptedImagesMessage(
            accepted_images=accepted_images,
            train_split=splits.train_percent,
            val_split=splits.val_percent,
        )

    def set_items_count(self, message: LabelingQueueAcceptedImagesMessage = None) -> None:
        """Set the number of items in the random splits table."""
        self._accepted_images = message.accepted_images
        self.gui.set_items_count(len(self._accepted_images))
        self.save_split_settings()
        splits = self.gui.get_split_settings()
        self.send_accepted_images_message(accepted_images=self._accepted_images, splits=splits)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
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
