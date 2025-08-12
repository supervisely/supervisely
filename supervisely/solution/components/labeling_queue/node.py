from typing import Callable, Dict, List, Optional, Tuple, Union
from venv import logger

from supervisely.api.api import Api
from supervisely.labeling_jobs.utils import Status
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.labeling_queue.automation import LabelingQueueRefresh
from supervisely.solution.components.labeling_queue.gui import LabelingQueueGUI
from supervisely.solution.engine.models import (
    LabelingQueueAcceptedImagesMessage,
    LabelingQueueRefreshInfoMessage,
    SampleFinishedMessage,
)


class LabelingQueueNode(SolutionElement):
    """
    LabelingQueue node for monitoring labeling tasks in a queue.
    """

    def __init__(
        self,
        queue_id: int,
        collection_id: int,
        x: int = 0,
        y: int = 0,
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

        # Initialize the base class before core blocks (to wrap publish/subscribe methods)
        super().__init__(*args, **kwargs)

        # --- core blocks --------------------------------------------------------
        self.gui = LabelingQueueGUI(queue_id=self.queue_id)
        self.node = SolutionCardNode(content=self.gui.card, x=x, y=y)
        self.modals = [self.gui.add_user_modal]

        self._setup_handlers()
        self.refresh_info()

        # automation after init (we need to wrap the publish methods in init first)
        self.automation = LabelingQueueRefresh(queue_id=self.queue_id, func=self.refresh_info)
        self.automation.apply(sec=self.REFRESH_INTERVAL_SEC)

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
    def refresh_info(
        self, message: Optional[SampleFinishedMessage] = None
    ) -> LabelingQueueRefreshInfoMessage:
        """
        Refresh the labeling queue info and update the GUI.
        """
        pending, annotating, reviewing, finished, rejected = 0, 0, 0, 0, 0
        try:
            pending, annotating, reviewing, finished, rejected = self.get_labeling_stats()
            self.gui.update_pending(pending)
            self.gui.update_annotation(annotating)
            self.gui.update_review(reviewing)
            self.gui.update_finished(finished)
            if finished > 0:
                self.get_new_accepted_images()
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
        return LabelingQueueAcceptedImagesMessage(accepted_images=img_ids.copy())

    def _available_subscribe_methods(self) -> Dict[str, Union[Callable, List[Callable]]]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "sample_finished": [self.add_items, self.refresh_info],
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "labeling_queue_info_refresh": self.refresh_info,
            "images_to_move": self.get_new_accepted_images,
        }

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
