from typing import Callable, Dict, List, Optional, Tuple, Union
from venv import logger

from supervisely.api.api import Api
from supervisely.labeling_jobs.utils import Status
from supervisely.solution.base_node import BaseQueueNode
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    LabelingQueuePerformanceMessage,
    LabelingQueueRefreshInfoMessage,
    SampleFinishedMessage,
)
from supervisely.solution.nodes.labeling_queue.automation import LabelingQueueRefresh
from supervisely.solution.nodes.labeling_queue.gui import LabelingQueueGUI


class LabelingQueueNode(BaseQueueNode):
    """
    LabelingQueue node for monitoring labeling tasks in a queue.
    """

    TITLE = "Labeling Queue"
    DESCRIPTION = "Labeling queue is a full annotation workflow where annotators pick the next available image from a shared queue. Once labeled, images are sent for review and quality check. Rejected images return to the same annotator."
    ICON = "mdi mdi-label-multiple"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        queue_id: int,
        collection_id: int,
        *args,
        **kwargs,
    ):
        """
        Initialize the LabelingQueue node.

        :param api: Supervisely API instance
        :param x: X coordinate of the node.
        :param y: Y Coordinate of the node.
        :param project_id: ID of the project to import data into.
        :param queue_id: ID of the labeling queue.
        """
        # --- parameters --------------------------------------------------------
        self.api = Api.from_env()
        self.queue_id = queue_id
        self.collection_id = collection_id
        self.REFRESH_INTERVAL_SEC = 30

        # --- core blocks --------------------------------------------------------
        self.gui = LabelingQueueGUI(queue_id=self.queue_id)
        self.modals = [self.gui.add_user_modal]

        # --- init node ------------------------------------------------------
        # * before automation (to wrap publish/subscribe methods)
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)

        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        self._setup_handlers()
        self.refresh_info()

        # automation after init (we need to wrap the publish methods in init first)
        self._automation = LabelingQueueRefresh(queue_id=self.queue_id)

    def _get_tooltip_buttons(self):
        return [self.gui.open_labeling_queue_btn]

    def configure_automation(self, *args, **kwargs):
        self._automation.apply(sec=self.REFRESH_INTERVAL_SEC, func=self.refresh_info)

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "sample_finished",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "queue_info_updated",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
            {
                "id": "train_val_split_items_count",
                "type": "source",
                "position": "left",
                "connectable": True,
                "style": {"top": "220"},
            },
            {
                "id": "labeling_performance",
                "type": "source",
                "position": "right",
                "connectable": True,
                "style": {"top": "18.555px"},
            },
        ]

    def _setup_handlers(self):
        """Setup handlers for buttons and other interactive elements"""

        @self.gui.add_annotator_btn.click
        def handle_add_annotator_btn_click():
            self.gui.annotators.show()
            self.gui.reviewers.hide()
            self.gui.add_user_modal.show()
            team_members = self.api.user.get_team_members(self.team_id)
            # filter out reviewers
            self.gui.user_selector.set(team_members)

        @self.gui.add_reviewer_btn.click
        def handle_add_reviewer_btn_click():
            self.gui.annotators.hide()
            self.gui.reviewers.show()
            self.gui.add_user_modal.show()
            team_members = self.api.user.get_team_members(self.team_id)
            # filter out annotators
            self.gui.user_selector.set(team_members)

    def get_json_data(self) -> dict:
        return {"queueId": self.queue_id}

    def get_labeling_stats(self):
        logger.info("Checking labeling queue info...")

        pending, annotating, reviewing, rejected, finished = 0, 0, 0, 0, 0
        queue_info = self.api.labeling_queue.get_info_by_id(self.queue_id)
        jobs = [self.api.labeling_job.get_info_by_id(job_id) for job_id in queue_info.jobs]
        completed = queue_info.status == Status.COMPLETED
        completed = completed or all(j.status == Status.COMPLETED for j in jobs)
        if completed:
            raise RuntimeError(
                f"Something went wrong: "
                f"Labeling queue {self.queue_id} is completed while it should be in progress."
            )

        finished += queue_info.accepted_count
        reviewing = self.api.labeling_queue.get_entities_count_by_status(self.queue_id, "done")
        annotating = queue_info.in_progress_count
        pending += queue_info.pending_count
        for job in jobs:
            for entity in job.entities:
                if entity["reviewStatus"] == "rejected":
                    rejected += 1

        logger.info(
            f"Labeling queue info: {self.queue_id}:\n"
            f"Pending: {pending}\n"
            f"Annotating: {annotating}\n"
            f"Reviewing: {reviewing}\n"
            f"Finished: {finished}\n"
            f"Rejected: {rejected}"
        )

        return pending, annotating, reviewing, finished, rejected

    def get_labeled_images_count(self) -> int:
        _, _, _, finished, _ = self.get_labeling_stats()
        return finished

    # ------------------------------------------------------------------
    # Events ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "project_updated": self.process_incoming_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "queue_info_updated": self.send_queue_info_updated_message,
            "train_val_split_items_count": self.send_new_items_message_to_train_val_split,
            "labeling_performance": self.send_performance_message,
        }

    def send_new_items_message_to_train_val_split(
        self, message: LabelingQueueAcceptedImagesMessage
    ):
        """Send message with all labeled images from labeling queue with status accepted"""
        return message

    def send_queue_info_updated_message(self, message: LabelingQueueRefreshInfoMessage):
        """Send message with all labeled images from labeling queue with status accepted"""
        return message

    def send_new_items_message(self):
        """Send message with all labeled images from labeling queue with status accepted"""
        images = self.get_new_accepted_images()
        msg = LabelingQueueAcceptedImagesMessage(accepted_images=images)
        self.send_new_items_message_to_train_val_split(msg)
        self.send_queue_info_updated_message(msg)

    def send_performance_message(self):
        """Send message to open labeling performance page"""
        return LabelingQueuePerformanceMessage(project_id=self.project_id)

    def process_incoming_message(self, message: SampleFinishedMessage):
        """Process incoming message from connected node."""
        if not isinstance(message, SampleFinishedMessage):
            raise TypeError("Expected SampleFinishedMessage, got {type(message)}")
        self.add_items(message)
        self.refresh_info(message)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def refresh_info(
        self, message: Optional[SampleFinishedMessage] = None
    ) -> LabelingQueueRefreshInfoMessage:
        """
        Refresh the labeling queue info and update the GUI.
        """
        pending, annotating, reviewing, finished, rejected = 0, 0, 0, 0, 0
        try:
            pending, annotating, reviewing, finished, rejected = self.get_labeling_stats()
            self.update_pending(pending)
            self.update_annotation(annotating)
            self.update_review(reviewing)
            self.update_finished(finished)
            if finished > 0:
                self.send_new_items_message()
        except Exception as e:
            logger.error(f"Failed to refresh labeling queue info: {str(e)}")
        return LabelingQueueRefreshInfoMessage(
            pending=pending,
            annotating=annotating,
            reviewing=reviewing,
            finished=finished,
            rejected=rejected,
        )

    def get_new_accepted_images(self) -> LabelingQueueAcceptedImagesMessage:
        """Get all labeled images from labeling queue with status accepted"""

        if not self.queue_id or not self.collection_id:
            return []

        resp = self.api.labeling_queue.get_entities_all_pages(
            self.queue_id,
            self.collection_id,
            status="accepted",
            filter_by=None,
        )

        img_ids = [entity["id"] for entity in resp["images"]]
        logger.info(f"Found {len(img_ids)} new accepted images in the labeling queue.")
        return img_ids

    def add_items(self, message: SampleFinishedMessage) -> None:
        """
        Add items to the labeling queue.
        :param message: Message containing the list of image IDs to add.
        """
        if not message.dst:
            logger.warning("No images to add to labeling queue.")
            return

        images = []
        for imgs in message.dst.values():
            images.extend(imgs)
        self.api.entities_collection.add_items(self.collection_id, images)
