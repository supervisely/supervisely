from typing import Callable, Dict, Optional

from supervisely.solution.components.project.node import ProjectNode
from supervisely.solution.engine.models import ImportFinishedMessage


class InputProjectNode(ProjectNode):
    IS_TRAINING = False
    TITLE = "Input Project"
    DESCRIPTION = "The Input Project is the central hub for all incoming data. Data in this project will not be modified."
    ICON = "mdi mdi-folder-home"
    ICON_COLOR = "#FFC40C"
    ICON_BG_COLOR = "#FFFFF0"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        super().__init__(
            *args,
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "import_finished",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "project_updated",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
            {
                "id": "embedding_status_request",
                "type": "source",
                "position": "left",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------

    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "import_finished": self.refresh,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "project_updated": self.send_project_updated_message,
            "embedding_status_request": self.send_check_embeddings_message,
        }

    def send_project_updated_message(self):
        pass

    def send_check_embeddings_message(self):
        pass

    def refresh(self, message: Optional[ImportFinishedMessage] = None) -> None:
        super().refresh(message)
        self.send_project_updated_message()
        self.send_check_embeddings_message()
