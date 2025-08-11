from typing import Callable, Dict, List, Literal, Optional, Tuple

from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.app.content import DataJson
from supervisely.app.widgets import Dialog
from supervisely.project.image_transfer_utils import move_structured_images
from supervisely.sly_logger import logger
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.move_labeled.automation import MoveLabeledAuto
from supervisely.solution.components.move_labeled.gui import MoveLabeledGUI
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    MoveLabeledDataFinishedMessage,
    TrainValSplitMessage,
)
from supervisely.solution.utils import get_interval_period


class MoveLabeledNode(SolutionElement):
    """
    This class is a placeholder for the MoveLabeled node.
    It is used to move labeled data from one location to another.
    """

    def __init__(
        self,
        src_project_id: int,
        dst_project_id: int,
        x: int = 0,
        y: int = 0,
        *args,
        **kwargs,
    ):
        self.api = Api.from_env()
        self.src_project_id = src_project_id
        self.dst_project_id = dst_project_id

        # --- core blocks --------------------------------------------------------
        self.automation = MoveLabeledAuto()
        self.gui = MoveLabeledGUI()
        self.card = self._build_card(
            title="Move Labeled Data",
            tooltip_description="Move labeled and accepted images to the Training Project.",
            icon="zmdi zmdi-dns",
            icon_color="#1976D2",
            icon_bg_color="#E3F2FD",
        )
        self.node = SolutionCardNode(content=self.card, x=x, y=y)

        # --- modals -------------------------------------------------------------
        self.modals = [self.modal]

        super().__init__(*args, **kwargs)

        self._images_to_move = []

        @self.card.click
        def on_automate_click():
            self.modal.show()

        @self.gui.automation_btn.click
        def on_automate_click():
            self.modal.hide()
            self.apply_automation(self.run)

        @self.gui.run_btn.click
        def on_run_click():
            self.modal.hide()
            self.node.show_in_progress_badge()
            self.run()
            self.node.hide_in_progress_badge()

    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "move_labeled_data_finished": self.run,
        }

    def _available_subscribe_methods(self):
        return {
            "images_to_move": self.set_images_to_move,
        }

    @property
    def modal(self):
        """
        Create the modal dialog for automation settings.
        """
        if not hasattr(self, "_modal"):
            self._modal = Dialog(
                title="Move Labeled Data",
                content=self.gui.widget,
            )
        return self._modal

    def _update_automation_details(self) -> Tuple[int, str, int, str]:
        enabled, _, _, min_batch, sec = self.gui.get_automation_details()
        if self.node is not None:
            period, interval = get_interval_period(sec)
            if enabled is not None:
                self.node.show_automation_badge()
                self.card.update_property("Run every", f"{interval} {period}", highlight=True)
                if min_batch is not None:
                    self.card.update_property("Min batch size", f"{min_batch}", highlight=True)
                else:
                    self.card.remove_property_by_key("Min batch size")
            else:
                self.node.hide_automation_badge()
                self.card.remove_property_by_key("Run every")
                self.card.remove_property_by_key("Min batch size")

    # publish event (may send Message object)
    def run(self) -> MoveLabeledDataFinishedMessage:
        if not self._images_to_move:
            logger.warning("No images to move. Returning empty message.")
            return MoveLabeledDataFinishedMessage(
                success=False,
                src={},
                dst={},
                items_count=0,
            )
        src, dst, total_moved = move_structured_images(
            self.api,
            self.src_project_id,
            self.dst_project_id,
            images=self._images_to_move,
        )
        return MoveLabeledDataFinishedMessage(
            success=True,
            src=src,
            dst=dst,
            items_count=total_moved,
        )

    def apply_automation(self, func: Callable[[], None], *args) -> None:
        """
        Apply the automation function to the MoveLabeled node.
        """
        enabled, _, _, _, sec = self.gui.get_automation_details()
        if not enabled:
            logger.warning("[MoveLabeledNode] Automation is not enabled.")
            sec = None
        self.automation.apply(func, sec, *args)
        self.save_settings()

    def save_settings(self) -> None:
        """
        Save the automation settings.
        This method is called when the user clicks the "Save settings" button.
        """
        self._update_automation_details()
        enabled, _, _, min_batch, sec = self.gui.get_automation_details()
        DataJson()[self.widget_id] = {"enabled": enabled, "sec": sec, "min_batch": min_batch}
        DataJson().send_changes()

    def load_settings(self) -> None:
        """
        Load the automation settings from DataJson.
        This method is called when the node is initialized.
        """
        data = DataJson().get(self.widget_id, {})
        enabled = data.get("enabled", False)
        sec = data.get("sec", 0)
        min_batch = data.get("min_batch", None)
        self.gui.update_automation_widgets(enabled, sec, min_batch)
        self._update_automation_details()

    def set_images_to_move(self, message: LabelingQueueAcceptedImagesMessage) -> None:
        """
        Set the images to move based on the message.
        """
        if not message.accepted_images:
            logger.warning("No items to move. Returning empty list.")
            self._images_to_move = []
            return

        self._images_to_move = message.accepted_images
        logger.info(f"Set {len(self._images_to_move)} images to move.")

