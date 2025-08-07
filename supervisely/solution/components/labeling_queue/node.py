from typing import Callable, List, Optional, Tuple
from venv import logger

from supervisely.api.api import Api
from supervisely.labeling_jobs.utils import Status
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.labeling_queue.automation import (
    LabelingQueueRefresh,
)
from supervisely.solution.components.labeling_queue.gui import LabelingQueueGUI


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
        self.api = Api.from_env()
        self.queue_id = queue_id
        self.collection_id = collection_id
        self._labeled_images = []

        self.gui = LabelingQueueGUI(queue_id=self.queue_id)
        self.automation = LabelingQueueRefresh(queue_id=self.queue_id, func=self.refresh_info)
        self.node = SolutionCardNode(content=self.gui.card, x=x, y=y)
        self.modals = [self.gui.add_user_modal]

        self._setup_handlers()
        self._callbacks: List[Callable[[], None]] = []
        self.apply_automation(sec=30)

        super().__init__(*args, **kwargs)

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
        return {
            # "projectId": self.project_id,
            "queueId": self.queue_id,
            "labeledImages": self._labeled_images,
        }

    def get_json_state(self) -> dict:
        return {}

    def update_pending(self, num: int):
        """Update number of pending images"""
        self.gui.update_pending(num)

    def update_annotation(self, num: int):
        """Update number of images being annotated"""
        self.gui.update_annotation(num)

    def update_review(self, num: int):
        """Update number of images being reviewed"""
        self.gui.update_review(num)

    def update_finished(self, num: int):
        """Update number of finished images"""
        self.gui.update_finished(num)

    def apply_automation(self, sec: int) -> None:
        """
        Apply the automation function to the MoveLabeled node.
        """
        self.automation.apply(sec=sec)

    def unschedule_refresh(self) -> None:
        """
        Unschedule the job that refreshes labeling queue info.
        """
        self.automation.unschedule_refresh()

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

    def refresh_info(self):
        """
        Refresh the labeling queue info and update the GUI.
        """
        try:
            pending, annotating, reviewing, finished, rejected = self.get_labeling_stats()
            self.update_pending(pending)
            self.update_annotation(annotating)
            self.update_review(reviewing)
            self.update_finished(finished)
            for callback in self._callbacks:
                callback()
        except RuntimeError as e:
            logger.error(str(e))
            self.unschedule_refresh()
            raise e
        except Exception as e:
            logger.error(f"Failed to refresh labeling queue info: {str(e)}")

    def on_refresh(self, func: Callable[[], None]) -> None:
        """
        Set a callback function to be called after refreshing the labeling queue info.
        :param callback: Function to call after refreshing.
        """
        self._callbacks.append(func)
        return func

    def get_new_accepted_images(self) -> List[int]:
        """Get all labeled images from labeling queue with status accepted"""

        if not self.queue_id or not self.collection_id:
            return []

        resp = self.api.labeling_queue.get_entities_all_pages(
            self.queue_id,
            self.collection_id,
            status="accepted",
            filter_by=None,
        )

        return [entity["id"] for entity in resp["images"]]
