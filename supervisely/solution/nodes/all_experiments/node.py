from typing import Callable, Dict, Literal, Optional, Union

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.io.env import project_id as env_project_id
from supervisely.solution.components.link_node.node import LinkNode
from supervisely.solution.engine.models import ReevaluateModelMessage, RegisterExperimentMessage


class AllExperimentsNode(LinkNode):
    """
    Node for displaying a link to the All Experiments page.
    """

    title = "All Experiments"
    description = "View all experiments with Training Project and explore their details."
    icon = "mdi mdi-flask-outline"
    icon_color = "#1976D2"
    icon_bg_color = "#E3F2FD"

    def __init__(
        self,
        project_id: int = None,
        width: int = 250,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        self._last_model = None
        link = "/nn/experiments"

        project_id = project_id or env_project_id()
        if project_id:
            link += f"?projects={project_id}"

        title = kwargs.pop("title", self.title)
        description = kwargs.pop("description", self.description)
        icon = kwargs.pop("icon", self.icon)
        icon_color = kwargs.pop("icon_color", self.icon_color)
        icon_bg_color = kwargs.pop("icon_bg_color", self.icon_bg_color)
        self._click_handled = True
        super().__init__(
            title=title,
            description=description,
            link=link,
            width=width,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            tooltip_position=tooltip_position,
            *args,
            **kwargs,
        )
        self._update_properties()

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "register_experiment",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
            {
                "id": "re_evaluate",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        return {
            "register_experiment": self._process_incomming_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "re_evaluate": self._send_model_to_evaluation,
        }

    def _process_incomming_message(self, message: RegisterExperimentMessage):
        self._set_last_model(message.model_path)

    def _send_model_to_evaluation(self) -> ReevaluateModelMessage:
        return ReevaluateModelMessage(model_path=self.last_model)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def last_model(self):
        return self._last_model

    @last_model.setter
    def last_model(self, value):
        """Set the last trained model."""
        self._last_model = value
        self._update_properties()

    def _set_last_model(self, model_path: str):
        """Set the last model for comparison."""
        if not isinstance(model_path, str):
            raise ValueError("Last model must be a string representing the model path.")
        if not model_path.startswith("/"):
            raise ValueError(
                "Last model should be from Team Files. Path must start with '/'. E.g. '/experiments/2730_my_project/48650_YOLO/checkpoints/best.pt'."
            )
        self._last_model = model_path
        self._update_properties()

    def _update_properties(self):
        """Update the node tooltip with the last model information."""
        if self.last_model is not None:
            model_name = self._last_model.split("/")[-1]
            link = (
                abs_url(f"/files/?path={self.last_model}")
                if is_development()
                else f"/files/?path={self.last_model}"
            )
            self.update_property(key="Last Model", value=model_name, link=link)
        else:
            self.remove_property_by_key(key="Last Model")
