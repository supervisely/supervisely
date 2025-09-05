from typing import Callable, Dict

from git import Optional

from supervisely.solution.components.project.node import ProjectNode
from supervisely.solution.engine.models import SampleFinishedMessage


class LabelingProjectNode(ProjectNode):
    IS_TRAINING = False
    TITLE = "Labeling Project"
    DESCRIPTION = "Project specifically for labeling data. All data in this project is in the labeling process. After labeling, data will be moved to the Training Project."
    ICON = "mdi mdi-folder-edit"
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

    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "sampling_finished": self.refresh,
            "pre_labeling_finished": self.refresh,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "project_updated": self.send_project_updated_message,
        }

    def _get_handles(self):
        return [
            {
                "id": "sampling_finished",
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
            {
                "id": "pre_labeling_finished",
                "type": "target",
                "position": "right",
                "label": "Output",
                "connectable": True,
            },
        ]

    def send_project_updated_message(self, message: Optional[SampleFinishedMessage] = None) -> None:
        """Sends a message indicating that the project has been updated."""
        return message

    def refresh(self, message: Optional[SampleFinishedMessage] = None) -> None:
        super().refresh(message)
        self.send_project_updated_message(message)
