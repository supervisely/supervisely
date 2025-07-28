from typing import Callable, List, Optional, Tuple
from venv import logger

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    Field,
    MembersListSelector,
    SolutionCard,
    SolutionGraph,
    Text,
    Widget,
)
from supervisely.labeling_jobs.utils import Status
from supervisely.solution.base_node import Automation, SolutionCardNode, SolutionElement
from supervisely.solution.scheduler import TasksScheduler


class LabelingQueueGUI:
    def __init__(self, queue_id: int):
        self.queue_id = queue_id

        self.pending_text = Text("Pending: 0 images")
        self.annotation_text = Text("Annotating: 0 images")
        self.review_text = Text("Reviewing: 0 images")
        self.finished_text = Text("Finished: 0 images")

        link = abs_url(f"labeling/jobs/list?queueId={self.queue_id}")
        self.open_labeling_queue_btn = Button(
            "Open Labeling Queue",
            icon="zmdi zmdi-open-in-new",
            button_size="mini",
            link=link,
            plain=True,
            button_type="text",
        )
        self.add_annotator_btn = Button(
            "Add annotator",
            icon="zmdi zmdi-accounts-add",
            button_size="mini",
            plain=True,
            button_type="text",
        )
        self.add_reviewer_btn = Button(
            "Add reviewer",
            icon="zmdi zmdi-account-add",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        self.user_selector = MembersListSelector(multiple=True)
        self.annotators = Field(
            title="Select annotators to add",
            content=self.user_selector,
        )
        self.annotators.hide()
        self.reviewers = Field(title="Select reviewers to add", content=self.user_selector)
        self.reviewers.hide()
        self.add_user_modal = Dialog(
            title="Add user",
            content=Container([self.annotators, self.reviewers], gap=0),
        )

        self.card = self._create_card()

    def _create_card(self) -> SolutionCard:
        return SolutionCard(
            title="Labeling Queue",
            tooltip=self._create_tooltip(),
            content=[self._create_nested_diagram()],
            width=200,
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        return SolutionCard.Tooltip(
            description="Labeling Queue management. Labeling queue is a full annotation workflow where annotators pick the next available image from a shared queue. Once labeled, images are sent for review and quality check. Rejected images return to the same annotator.",
            content=[
                self.open_labeling_queue_btn,
                self.add_annotator_btn,
                self.add_reviewer_btn,
            ],
        )

    def _create_nested_diagram(self) -> SolutionGraph:
        styles = "margin: 0; box-shadow: none; background-color: #F0F4F8; padding: 9px; width: 165px; border-radius: 6px; box-sizing: border-box;"
        pending_cont = Container([self.pending_text], style=styles)
        annotation_cont = Container([self.annotation_text], style=styles)
        review_cont = Container([self.review_text], style=styles)
        finished_cont = Container([self.finished_text], style=styles)

        node_pending = SolutionGraph.Node(
            x=0,
            y=0,
            content=pending_cont,
            width=165,
            padding=2,
        )
        node_annotation = SolutionGraph.Node(
            x=0,
            y=70,
            content=annotation_cont,
            width=165,
            padding=2,
        )
        node_review = SolutionGraph.Node(
            x=0,
            y=140,
            content=review_cont,
            width=165,
            padding=2,
        )
        node_finished = SolutionGraph.Node(
            x=0,
            y=220,
            content=finished_cont,
            width=165,
            padding=2,
        )

        nested_nodes = [node_pending, node_annotation, node_review, node_finished]
        nested_connections = {
            node_pending.key: [
                [node_annotation.key, {"path": "straight", "size": 1, "dash": {"len": 8, "gap": 8}}]
            ],
            node_annotation.key: [
                [node_review.key, {"path": "straight", "size": 1, "dash": {"len": 8, "gap": 8}}]
            ],
            node_review.key: [
                [
                    node_finished.key,
                    {
                        "color": "#00A141",  # "#9ee1ac",
                        "middleLabel": "Accepted",
                        "fontSize": 10,
                        "size": 1,
                        "dash": {"len": 8, "gap": 8},
                    },
                ],
                [
                    node_annotation.key,
                    {
                        "color": "#F31d1d",  # "#fecbcb",
                        "startSocket": "right",
                        "endSocket": "right",
                        "middleLabel": "Rejected",
                        "fontSize": 10,
                        "path": "grid",
                        "size": 1,
                        # "dash": {"len": 8, "gap": 8},
                    },
                ],
            ],
        }
        return SolutionGraph(
            nodes=nested_nodes,
            connections=nested_connections,
            height="275px",
            width="100%",
        )

    def update_pending(self, num: int):
        self.pending_text.text = f"Pending: {num} images"

    def update_annotation(self, num: int):
        self.annotation_text.text = f"Annotating: {num} images"

    def update_review(self, num: int):
        self.review_text.text = f"Reviewing: {num} images"

    def update_finished(self, num: int):
        self.finished_text.text = f"Finished: {num} images"


class LabelingQueueRefresh(Automation):
    """
    Automation for refreshing labeling queue information periodically
    """

    def __init__(self, queue_id: int, func: Optional[Callable[[], None]] = None):
        super().__init__()
        self.job_id = f"refresh_labeling_queue_{queue_id}"
        self.queue_id = queue_id
        self.func = func

    def apply(self, sec: int) -> None:
        self.scheduler.add_job(self.func, interval=sec, job_id=self.job_id, replace_existing=True)

    def schedule_refresh(self, func: Callable[[], None], interval_sec: int = 5) -> None:
        """
        Schedule a job to refresh labeling queue info.
        """
        self.scheduler.add_job(
            func, interval=interval_sec, job_id=self.job_id, replace_existing=True
        )
        logger.info(
            f"Scheduled refresh for labeling queue {self.queue_id} every {interval_sec} seconds"
        )

    def unschedule_refresh(self) -> None:
        """
        Unschedule the job that refreshes labeling queue info.
        """
        if self.scheduler.is_job_scheduled(self.job_id):
            self.scheduler.remove_job(self.job_id)
            logger.info(f"Unscheduled refresh for labeling queue {self.queue_id}")


class LabelingQueue(SolutionElement):
    """
    LabelingQueue node for monitoring labeling tasks in a queue.
    """

    def __init__(
        self,
        api: Api,
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
        self.api = api
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
