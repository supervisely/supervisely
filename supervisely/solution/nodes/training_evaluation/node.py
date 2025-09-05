from typing import Optional

from supervisely._utils import abs_url, is_development
from supervisely.sly_logger import logger
from supervisely.solution.components.link_node.node import LinkNode
from supervisely.solution.engine.models import TrainFinishedMessage


class TrainingEvaluationReportNode(LinkNode):
    """Node for linking to the training session evaluation page."""

    TITLE = "Evaluation Report"
    DESCRIPTION = "Open the training session evaluation page to explore detailed evaluation metrics and visualizations for the trained model. Evaluation was performed on the validation set after training completion."
    ICON = "mdi mdi-chart-line-variant"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(self, project_id: Optional[int] = None, *args, **kwargs):
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        link = f"/projects/{project_id}/stats/datasets" if project_id is not None else None
        link = kwargs.pop("link", link)

        self.project_id = project_id
        self._click_handled = True
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            link=link,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "training_finished",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self):
        return {"training_finished": self._process_incoming_message}


    def _process_incoming_message(self, message: TrainFinishedMessage):
        if not hasattr(message, "experiment_info"):
            logger.warning("Received message does not have 'experiment_info' attribute.")
            return
        evaluation_report_id = message.experiment_info.get("evaluation_report_id")
        self.set_report(evaluation_report_id)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_report(self, evaluation_report_id: Optional[int] = None):
        """Receive experiment_info and set link to evaluation report by ID."""
        if evaluation_report_id is None:
            self.remove_badge_by_key("Evaluation Report")
            self.remove_property_by_key("Evaluation Report")
            self.remove_link()
            return

        link = f"/model-benchmark?id={evaluation_report_id}"
        if is_development():
            link = abs_url(link)
        
        self.update_badge_by_key(key="Evaluation Report", label="new", badge_type="success")
        self.update_property("Evaluation Report", "open 🔗", link=link)
        self.set_link(link)
