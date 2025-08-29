from typing import Callable, Dict

from supervisely.solution.components.project.node import ProjectNode


class TrainingProjectNode(ProjectNode):
    is_training = True
    title = "Training Project"
    description = "Project specifically for training data. All data in this project is in the training process. After training, data will be moved to the Training Project."
    icon = "mdi mdi-folder-multiple-image"
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
            "train_val_split_finished": self.refresh,
        }

    def _available_publish_methods(self):
        return {
            "qa_stats": self.send_message_to_qa_stats,
            "data_versioning_project_id": self.send_message_to_data_versioning,
        }

    def send_message_to_qa_stats(self) -> None:
        pass

    def send_message_to_data_versioning(self) -> None:
        pass

    def _get_handles(self):
        return [
            {
                "id": "train_val_split_finished",
                "type": "target",
                "position": "top",
                "label": "Input",
                "connectable": True,
            },
            {
                "id": "qa_stats_project_id",
                "type": "source",
                "position": "right",
                "label": "QA & Stats",
                "connectable": True,
                "style": {"top": "18.555px"},
            },
            {
                "id": "data_versioning_project_id",
                "type": "source",
                "position": "bottom",
                "label": "Data Versioning",
                "connectable": True,
            },
        ]
