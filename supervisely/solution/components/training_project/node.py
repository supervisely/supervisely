from typing import Callable, Dict

from supervisely.solution.components.project.node import ProjectNode


class TrainingProjectNode(ProjectNode):
    is_training = True
    title = "Training Project"
    description = (
        "Project specifically for training data. All data in this project is in the training process. After training, data will be moved to the Training Project."
    )

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
            "train_val_split_finished": self.update,
        }

    def _available_handles(self):
        return [
            {
                "id": "target-1",
                "type": "target",
                "position": "top",
                "label": "Input",
                "connectable": True,
            },
            {
                "id": "target-2",
                "type": "target",
                "position": "left",
                "label": "Input",
                "connectable": True,
            },
            {
                "id": "source-1",
                "type": "source",
                "position": "bottom",
                "label": "Output",
                "connectable": True,
            },
            {
                "id": "target-2",
                "type": "source",
                "position": "right",
                "label": "Output",
                "connectable": True,
                "style": {"top": "20px"},
            },
            {
                "id": "source-3",
                "type": "source",
                "position": "right",
                "label": "Output",
                "connectable": True,
                "style": {"top": "70px"},
            },
        ]
