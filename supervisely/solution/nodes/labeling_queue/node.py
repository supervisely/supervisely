import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from venv import logger

from supervisely._utils import batched
from supervisely.api.api import Api
from supervisely.io.env import team_id as env_team_id
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
from supervisely.solution.utils import find_agent


class LabelingQueueNode(BaseQueueNode):
    """
    LabelingQueue node for monitoring labeling tasks in a queue.
    """

    TITLE = "Labeling Queue"
    DESCRIPTION = "Labeling Queue management. Labeling Queue is a workflow where annotators pick the next available image from a shared queue. Once labeled, images are sent for review. Rejected images return to the same annotator."
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
        # self.modals = [self.gui.add_user_modal]
        self.modals = []

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

        @self.gui.debug_btn_1.click
        def handle_debug_btn_1_click():
            self._debug_1()

        @self.gui.debug_btn_2.click
        def handle_debug_btn_2_click():
            self._debug_2()

    def _get_tooltip_buttons(self):
        return [self.gui.open_labeling_queue_btn, self.gui.debug_btn_1, self.gui.debug_btn_2]

    def configure_automation(self, *args, **kwargs):
        self._automation.apply(sec=self.REFRESH_INTERVAL_SEC, func=self.refresh_info)

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "sampling_finished",
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

        # @self.gui.add_annotator_btn.click
        # def handle_add_annotator_btn_click():
        #     # self.gui.annotators.show()
        #     # self.gui.reviewers.hide()
        #     # self.gui.add_user_modal.show()
        #     team_members = self.api.user.get_team_members(self.team_id)
        #     # filter out reviewers
        #     # self.gui.user_selector.set(team_members)

        # @self.gui.add_reviewer_btn.click
        # def handle_add_reviewer_btn_click():
        #     # self.gui.annotators.hide()
        #     # self.gui.reviewers.show()
        #     # self.gui.add_user_modal.show()
        #     team_members = self.api.user.get_team_members(self.team_id)
        #     # filter out annotators
        #     # self.gui.user_selector.set(team_members)

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
            "accepted_images": self.send_accepted_images_message,
            "labeling_performance": self.send_performance_message,
        }

    def send_accepted_images_message(self):
        """Send message with all labeled images from labeling queue with status accepted"""
        images = self.get_new_accepted_images()
        return LabelingQueueAcceptedImagesMessage(accepted_images=images)

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
                self.send_accepted_images_message()
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

    # ------------------------------------------------------------------
    # -- DEBUG ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _debug(self, annotate: bool = False):
        def _get_next(api: Api, job_id: int):
            while True:
                try:
                    resp = api.post("labeling-queues.entities.get-next", {"id": job_id})
                    resp_json = resp.json()
                    entity_id = resp_json.get("id")
                    if entity_id is None:
                        break
                    yield entity_id
                except Exception as e:
                    break

        def _get_queues_job(api: Api, queue_info):
            ann_job, review_job = None, None
            for job in queue_info.jobs:
                job_info = api.labeling_job.get_info_by_id(job)
                job_type = job_info.name.lower().split(" ")
                if "annotation" in job_type:
                    ann_job = job_info
                elif "review" in job_type:
                    review_job = job_info
            return ann_job, review_job

        self.update_badge_by_key(key="Labeling", label="in progress", badge_type="info")
        queue_info = self.api.labeling_queue.get_info_by_id(self.queue_id)
        if queue_info.status == self.api.labeling_job.Status.COMPLETED.value:
            logger.info(f"Labeling queue {self.queue_id} is already completed")
            return

        all_items = self.api.labeling_queue.get_entities_all_pages(self.queue_id).get("images", [])
        all_items = [item.get("id") for item in all_items]
        if annotate:
            checkpoint = (
                "/yolov8_train/object detection/Train Insulator Dataset/4842/weights/best_93.pt"
            )
            agent_id = find_agent(self.api, env_team_id())
            # model = self.api.nn.connect(51017)
            model = self.api.nn.deploy(checkpoint, agent_id=agent_id)
            for batch in batched(all_items, 16):
                model.predict(image_id=batch, upload_mode="iou_merge")
            time.sleep(1)
            model.shutdown()

        ann_job, review_job = _get_queues_job(self.api, queue_info)

        for entity_id in _get_next(self.api, ann_job.id):
            self.api.labeling_job.set_entity_review_status(ann_job.id, entity_id, "accepted")

        for entity_id in _get_next(self.api, review_job.id):
            self.api.labeling_job.set_entity_review_status(review_job.id, entity_id, "accepted")

        self.refresh_info()
        self.remove_badge_by_key(key="Labeling")

    def _debug_1(self):
        self._debug(annotate=True)

    def _debug_2(self):
        self._debug(annotate=False)
