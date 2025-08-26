from typing import Callable, Dict

from supervisely.solution.components.project.node import ProjectNode


class LabelingProjectNode(ProjectNode):
    is_training = False
    title = "Labeling Project"
    description = "Project specifically for labeling data. All data in this project is in the labeling process. After labeling, data will be moved to the Training Project."

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

    def send_project_updated_message(self) -> None:
        """Sends a message indicating that the project has been updated."""
        pass
