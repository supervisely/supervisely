from typing import Optional

from supervisely.sly_logger import logger
from supervisely.solution.components.link_node.node import LinkNode
from supervisely.solution.engine.models import TrainFinishedMessage


class TrainingMetricsNode(LinkNode):
    """Node for linking to the training session metrics page."""

    TITLE = "Training Metrics"
    DESCRIPTION = "Open the training session metrics page."
    ICON = "mdi mdi-chart-line-variant"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(self, project_id: Optional[int] = None, *args, **kwargs):
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        link = f"/projects/{project_id}/stats/datasets" if project_id is not None else ""
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
                "id": "train_finished",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self):
        return {"train_finished": self._process_incoming_message}

    def _process_incoming_message(self, message: TrainFinishedMessage):
        if not hasattr(message, "experiment_info"):
            logger.warning("Received message does not have 'experiment_info' attribute.")
            return

        logs_info = message.experiment_info.get("logs")
        logs_link = logs_info.get("link")
        self.set_metrics(logs_link)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_metrics(self, logs_link: str = None):
        """Receive experiment_info and set link to logs.dir."""
        if logs_link is not None:
            link = logs_link
        else:
            link = None

        if link is not None:
            self.update_badge_by_key(key="status", value="Metrics", badge_type="success")
            self.update_property("Metrics Link", "Open Metrics", link=link, highlight=True)
            self.set_link(link)
        else:
            self.remove_badge_by_key("status")
            self.remove_property_by_key("Metrics Link")
            self.remove_link()
