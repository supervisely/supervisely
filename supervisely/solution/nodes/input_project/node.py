from typing import Callable, Dict, Optional

from supervisely.solution.components.project.node import ProjectNode
from supervisely.solution.engine.models import ImportFinishedMessage


class InputProjectNode(ProjectNode):
    is_training = False
    title = "Input Project"
    description = "The Input Project is the central hub for all incoming data. Data in this project will not be modified."
    icon = "mdi mdi-folder-home"
    icon_color = "#FFC40C"
    icon_bg_color = "#FFFFF0"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        title = kwargs.pop("title", self.title)
        description = kwargs.pop("description", self.description)
        super().__init__(
            *args,
            title=title,
            description=description,
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
            "import_finished": self.update,
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

    def update(self, message: Optional[ImportFinishedMessage] = None) -> None:
        super().update(message)
        self.send_project_updated_message()
        self.send_check_embeddings_message()
