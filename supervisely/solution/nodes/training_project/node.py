from typing import Callable, Dict

from supervisely.solution.components.project.node import ProjectNode


class TrainingProjectNode(ProjectNode):
    IS_TRAINING = True
    TITLE = "Training Project"
    DESCRIPTION = "Project specifically for training data. All data in this project is in the training process. After training, data will be moved to the Training Project."
    ICON = "mdi mdi-folder-multiple-image"
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
            "train_val_split_finished": self.refresh,
            "add_training_data_id": self.refresh,
            "move_labeled_data_finished": self.refresh,
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

    def send_message_to_add_training_data(self) -> None:
        pass

    def _get_handles(self):
        return [
            {
                "id": "move_labeled_data_finished",
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
            {
                "id": "add_training_data_id",
                "type": "source",
                "position": "left",
                "label": "Add Training Data",
                "connectable": True,
            },
        ]
