from typing import Optional

from supervisely.solution.components.link_node.node import LinkNode
from supervisely.solution.engine.models import TrainFinishedMessage


class TrainingExperimentNode(LinkNode):
    """Node for linking to the training experiment."""

    TITLE = "Training Experiment"
    DESCRIPTION = "Link to the training experiment."
    ICON = "mdi mdi-file-star"
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
        return {"train_finished": self.set_experiment_link}

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_experiment_link(self, message: TrainFinishedMessage):
        """Receive experiment_info and set link to experiment by experiment_id."""
        try:
            experiment_info = message.experiment_info or {}
            experiment_id = experiment_info.get("experiment_id")
            if experiment_id is not None:
                link = f"/nn/experiments/{experiment_id}"
                self.set_link(link)
        except Exception:
            # Silently ignore malformed message
            pass
