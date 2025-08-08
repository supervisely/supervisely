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
from supervisely.solution.engine.models import MoveLabeledDataFinishedMessage
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
        )
        self.node = SolutionCardNode(content=self.card, x=x, y=y)

        # --- modals -------------------------------------------------------------
        self.modals = [self.modal]

        super().__init__(*args, **kwargs)

        @self.card.click
        def on_automate_click():
            self.modal.show()

        # @self.gui.automation_btn.click
        # def on_automate_click():
        #     self.apply_automation(self._get_new_images)

    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {"move_labeled_data_finished": self.run}

    @property
    def modal(self):
        """
        Create the modal dialog for automation settings.
        """
        if not hasattr(self, "_modal"):
            self._modal = Dialog(
                title="Move Labeled Data",
                size="tiny",
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
    def run(self, image_ids: Dict[int, List[ImageInfo]]) -> MoveLabeledDataFinishedMessage:
        src, dst, total_moved = move_structured_images(
            self.api,
            self.src_project_id,
            self.dst_project_id,
            images=image_ids,
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
        _, _, _, _, sec = self.gui.get_automation_details()
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

    def add_to_collection(
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

        main_collection_name = f"All_{split_name}"
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
