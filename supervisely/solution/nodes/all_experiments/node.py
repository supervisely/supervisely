from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple, Union

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.io.env import project_id as env_project_id
from supervisely.solution.components.link_node.node import LinkNode
from supervisely.solution.engine.models import (
    ComparisonFinishedMessage,
    TrainingFinishedMessage,
)


class AllExperimentsNode(LinkNode):
    """
    Node for displaying a link to the All Experiments page.
    """

    TITLE = "All Experiments"
    DESCRIPTION = "Track all experiments in one place. The best model for comparison will be selected from the list of experiments based on the primary metric (e.g. for detection tasks, the primary metric is mAP)."
    ICON = "mdi mdi-flask-outline"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        project_id: int = None,
        width: int = 250,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        self._api = Api.from_env()
        self._best_model = None
        self._best_model_task_id = None
        self._project_id = project_id

        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        self._click_handled = True
        super().__init__(
            title=title,
            description=description,
            width=width,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            tooltip_position=tooltip_position,
            *args,
            **kwargs,
        )
        # self._update_properties()

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
                "id": "best_model",
                "type": "target",
                "position": "right",
                "connectable": True,
            },
            {
                "id": "experiment_registered",
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
            "register_experiment": self._process_training_finished_message,
            "best_model": self._process_comparison_finished_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "experiment_registered": self._send_model_to_evaluation,
        }

    def _process_comparison_finished_message(self, message: ComparisonFinishedMessage):
        self._best_model_task_id = message.train_task_id
        self._update_properties()

    def _process_training_finished_message(self, message: TrainingFinishedMessage):
        if self._project_id is None:
            self._project_id = self._extract_project_id(message.task_id)
            if self._project_id is not None:
                self._update_link(self._project_id)
        if self._best_model_task_id is not None:
            self._send_model_to_evaluation(self._best_model_task_id)

    def _send_model_to_evaluation(self, task_id: int) -> TrainingFinishedMessage:
        return TrainingFinishedMessage(task_id=task_id)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _update_link(self, project_id: Optional[int] = None):
        """Set project ID and update the link accordingly."""
        link = "/nn/experiments"
        project_id = project_id or self._project_id
        if project_id is not None:
            link += f"?projects={project_id}"
        if is_development():
            link = abs_url(link)
        self.set_link(link)

    def _extract_project_id(self, task_id: int) -> Optional[int]:
        """Extract project ID from the task ID."""
        task_info = self._api.task.get_info_by_id(task_id)
        experiment = task_info.get("meta", {}).get("output", {}).get("experiment", {}).get("data")
        if isinstance(experiment, dict):
            return experiment.get("project_id")

    def _update_properties(self):
        if self._best_model_task_id is None:
            return
        experiment = self._extract_experiment_info(self._best_model_task_id)
        metric_name = self._parse_primary_metric_name(experiment)
        metric_value = self._parse_primary_metric_value(experiment)
        checkpoint_path = self._parse_checkpoint_link(experiment)
        checkpoint_name = Path(checkpoint_path).name if checkpoint_path else None

        self.update_property(key="Task ID", value=str(self._best_model_task_id))
        if metric_name and metric_value is not None:
            self.update_property(key=metric_name, value=str(round(metric_value, 4)), highlight=True)
        if checkpoint_name and checkpoint_path:
            self.update_property(key="Best Checkpoint", value=checkpoint_name, link=checkpoint_path)

    def _extract_experiment_info(self, task_id: int) -> Optional[float]:
        """Extract the primary metric value from the task info."""
        task_info = self._api.task.get_info_by_id(task_id)
        if not task_info:
            return None
        experiment = task_info.get("meta", {}).get("output", {}).get("experiment", {}).get("data")
        return experiment

    def _parse_primary_metric_name(
        self,
        experiment: dict,
    ) -> Optional[float]:
        """Extract the primary metric value from the experiment info."""
        if not experiment:
            return None
        return experiment.get("primary_metric")

    def _parse_primary_metric_value(self, experiment: dict) -> Optional[float]:
        """Extract the primary metric value from the experiment info."""
        if not experiment:
            return None
        primary_metric = experiment.get("primary_metric")
        metrics = experiment.get("evaluation_metrics", {})
        return metrics.get(primary_metric)

    def _parse_checkpoint_link(self, experiment: dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract the best checkpoint path from the experiment info."""
        if not experiment:
            return None
        artifacts_dir = experiment.get("artifacts_dir")
        best_checkpoint = experiment.get("best_checkpoint")
        if artifacts_dir and best_checkpoint:
            checkpoint_path = str(Path(artifacts_dir) / "checkpoints" / best_checkpoint)
            checkpoint_path = f"/files/?path={checkpoint_path}"
            if is_development():
                checkpoint_path = abs_url(checkpoint_path)
            return checkpoint_path
        return None
