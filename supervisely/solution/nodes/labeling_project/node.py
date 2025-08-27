from typing import Callable, Dict

from git import Optional

from supervisely.solution.components.project.node import ProjectNode
from supervisely.solution.engine.models import SampleFinishedMessage


class LabelingProjectNode(ProjectNode):
    is_training = False
    title = "Labeling Project"
    description = "Project specifically for labeling data. All data in this project is in the labeling process. After labeling, data will be moved to the Training Project."
    icon = "mdi mdi-folder-edit"
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

    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "sample_finished": self.update,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "project_updated": self.send_project_updated_message,
        }

    def _available_handles(self):
        return [
            {
                "id": "sample_finished",
                "type": "target",
                "position": "top",
                "label": "Input",
                "connectable": True,
            },
            {
                "id": "project_updated",
                "type": "source",
                "position": "bottom",
                "label": "Output",
                "connectable": True,
            },
        ]

    def send_project_updated_message(self, message: Optional[SampleFinishedMessage] = None) -> None:
        """Sends a message indicating that the project has been updated."""
        return message

    def update(self, message: Optional[SampleFinishedMessage] = None) -> None:
        super().update(message)
        self.send_project_updated_message(message)
