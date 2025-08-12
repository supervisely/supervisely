from typing import Callable, Dict, List, Literal, Optional, Tuple

from supervisely.api.api import Api
from supervisely.project.image_transfer_utils import move_structured_images
from supervisely.sly_logger import logger
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.move_labeled.automation import MoveLabeledAuto
from supervisely.solution.components.move_labeled.gui import MoveLabeledGUI
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    MoveLabeledDataFinishedMessage,
)


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
        # First, initialize the base class (to wrap publish/subscribe methods)
        super().__init__(*args, **kwargs)

        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.src_project_id = src_project_id
        self.dst_project_id = dst_project_id
        self._images_to_move = []

        # --- core blocks --------------------------------------------------------
        self.automation = MoveLabeledAuto(self.run)
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
        self.modals = [self.gui.modal, self.automation.modal]

        @self.card.click
        def on_automate_click():
            self.gui.modal.show()

        @self.automation.apply_button.click
        def on_automate_click():
            self.automation.modal.hide()
            self.apply_automation()

        @self.gui.run_btn.click
        def on_run_click():
            self.gui.modal.hide()
            self.node.show_in_progress_badge()
            self.run()
            self.node.hide_in_progress_badge()

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "move_labeled_data_finished": self.run,
        }

    def _available_subscribe_methods(self):
        """Returns a dictionary of methods that can be used as callbacks for subscribed events."""
        return {
            "images_to_move": self.set_images_to_move,
        }

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

    # subscribe event (may receive Message object)
    def set_images_to_move(self, message: LabelingQueueAcceptedImagesMessage) -> None:
        """
        Set the images to move based on the message.
        """
        if not message.accepted_images:
            logger.warning("No items to move. Returning empty list.")
            self._images_to_move = []
        else:
            self._images_to_move = message.accepted_images
        logger.info(f"Set {len(self._images_to_move)} images to move.")
        self.gui.set_items_count(len(self._images_to_move))
        self.card.update_property("Available items to move", f"{len(self._images_to_move)}")

    # ------------------------------------------------------------------
    # Automation ---------------------------------------------------
    # ------------------------------------------------------------------
    def update_automation_details(self) -> Tuple[int, str, int, str]:
        enabled, period, interval, min_batch, sec = self.automation.get_details()
        if self.node is not None:
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

    def apply_automation(self) -> None:
        """
        Apply the automation function to the MoveLabeled node.
        """
        self.automation.apply()
        self.update_automation_details()
