import random
from typing import Callable, Dict, List, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.app.content import DataJson
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    MoveLabeledDataFinishedMessage,
)
from supervisely.solution.nodes.add_training_data.gui import (
    AddTrainingDataGUI,
)


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

        # --- core blocks --------------------------------------------------------
        self.gui = AddTrainingDataGUI()
        self.modal_content = self.gui.widget

        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.dst_project_id = dst_project_id
        self._click_handled = True

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
            # self.save_split_settings(settings_data)
            pass

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
            "training_data_added": self.copy_data,
        }

    def _available_publish_methods(self):
        """Returns a dictionary of methods that can be used for publishing events."""
        return {"train_val_split_finished": self.send_data_copied_message}
    
    def send_data_copied_message(self):
        pass

    def copy_data(self) -> MoveLabeledDataFinishedMessage:
        """
        """

        return MoveLabeledDataFinishedMessage()