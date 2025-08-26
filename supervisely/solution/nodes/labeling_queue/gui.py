from typing import Callable, List, Optional, Tuple
from venv import logger

from supervisely._utils import abs_url
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    Field,
    MembersListSelector,
    SolutionCard,
    SolutionGraph,
    Text,
    Icons,
)


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
            icon=Icons(
                class_name="zmdi zmdi-brush",
                color="#1976D2",
                bg_color="#E3F2FD",
            )
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        return SolutionCard.Tooltip(
            description="Labeling Queue management. Labeling queue is a full annotation workflow where annotators pick the next available image from a shared queue. Once labeled, images are sent for review and quality check. Rejected images return to the same annotator.",
            content=[
                self.open_labeling_queue_btn,
                # self.add_annotator_btn,
                # self.add_reviewer_btn,
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
