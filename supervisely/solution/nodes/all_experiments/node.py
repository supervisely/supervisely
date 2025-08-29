from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Union

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.io.env import project_id as env_project_id
from supervisely.solution.components.link_node.node import LinkNode
from supervisely.solution.engine.models import TrainingFinishedMessage


class AllExperimentsNode(LinkNode):
    """
    Node for displaying a link to the All Experiments page.
    """

    TITLE = "All Experiments"
    DESCRIPTION = "View all experiments with Training Project and explore their details."
    ICON = "mdi mdi-flask-outline"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        project_id: int = None,
        width: int = 250,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        self._api = Api.from_env()
        self._best_model = None
        self._best_model_task_id = None
        self._project_id = project_id or env_project_id()
        self._last_task_id = None
        # self._update_link()

        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        self._click_handled = True
        super().__init__(
            title=title,
            description=description,
            width=width,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            tooltip_position=tooltip_position,
            *args,
            **kwargs,
        )
        # self._update_properties()

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
            {
                "id": "training_finished",
                "type": "source",
                "position": "right",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        return {
            "training_finished": self._process_incomming_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "training_finished": self._send_model_to_evaluation,
        }

    def _process_incomming_message(self, message: TrainingFinishedMessage):
        project_id = self._extract_project_id(message.task_id)
        if project_id is not None:
            self._update_link(project_id)
        if self._last_task_id is not None:
            self._send_model_to_evaluation(message.task_id)
        self._last_task_id = message.task_id

    def _send_model_to_evaluation(self, task_id: int) -> TrainingFinishedMessage:
        return TrainingFinishedMessage(task_id=task_id)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _update_link(self, project_id: Optional[int] = None):
        """Set project ID and update the link accordingly."""
        link = "/nn/experiments"
        project_id = project_id or self._project_id
        if project_id is not None:
            link += f"?projects={project_id}"
        if is_development():
            link = abs_url(link)
        self.set_link(link)

    def _extract_project_id(self, task_id: int) -> Optional[int]:
        """Extract project ID from the task ID."""
        task_info = self._api.task.get_info_by_id(task_id)
        experiment = task_info.get("meta", {}).get("output", {}).get("experiment", {}).get("data")
        if isinstance(experiment, dict):
            return experiment.get("project_id")
