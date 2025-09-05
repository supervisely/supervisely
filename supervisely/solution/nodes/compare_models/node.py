import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import supervisely.io.env as sly_env
from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    ComparisonFinishedMessage,
    EvaluationFinishedMessage,
    TrainingFinishedMessage,
)
from supervisely.solution.nodes.compare_models.gui import ComparisonGUI
from supervisely.solution.nodes.compare_models.history import ComparisonHistory


class CompareModelsNode(BaseCardNode):
    APP_SLUG = "supervisely-ecosystem/model-benchmark/compare_models"
    TITLE = "Compare Models"
    DESCRIPTION = "Compare evaluation results from the latest training session against the best model reference report. Helps track performance improvements over time and identify the most effective training setups. If the new model performs better, it can be used to re-deploy the NN model for pre-labeling to speed-up the process."
    ICON = "mdi mdi-compare"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        project_id: Optional[int] = None,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        """A node for comparing evaluation reports of different models in Supervisely."""
        self._api = Api.from_env()
        project_id = project_id or sly_env.project_id()
        self.project = self._api.project.get_info_by_id(project_id)
        self.team_id = self.project.team_id
        self.workspace_id = self.project.workspace_id

        self.history = ComparisonHistory()
        self.gui = ComparisonGUI(team_id=self.team_id)

        # --- init node ------------------------------------------------------
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        self.modal_content = self.gui.content
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        # last experiment info:
        self._last_experiment_task_id = None
        self._last_experiment_eval_dir = None

        # best experiment info:
        self._best_experiment_eval_dir = None
        self._best_experiment_task_id = None

        @self.gui.automation_switch.value_changed
        def on_automation_switch_change(value: bool):
            self.save(enabled=value)

        @self.gui.agent_selector.value_changed
        def on_agent_selector_change(value: int):
            self.save(agent_id=value)

        @self.gui.run_btn.click
        def run_comparison():
            self.run()

        @self.click
        def on_node_click():
            self.gui.settings_modal.show()

        self.modals = [self.history.modal, self.gui.settings_modal]

        self._update_automation_properties(self.gui.automation_switch.is_switched())

    # ------------------------------------------------------------------
    # Node methods -----------------------------------------------------
    # ------------------------------------------------------------------
    def _get_tooltip_buttons(self):
        if not hasattr(self, "tooltip_buttons"):
            self.tooltip_buttons = [self.gui.run_btn, self.history.history_btn]
        return self.tooltip_buttons

    def _update_automation_properties(self, enable: bool):
        """Update node properties with current settings."""
        value = "enabled" if enable else "disabled"
        self.update_property("Compare models", value, highlight=enable)
        if enable:
            self.show_automation_badge()
        else:
            self.hide_automation_badge()

    def _update_best_model_properties(
        self,
        best_checkpoint: str,
        best_metric: float,
        metric_name: str,
        task_id: int,
    ):
        """Update node properties with current best model info."""
        if not best_checkpoint.startswith("/files"):
            best_checkpoint = f"/files/?path={best_checkpoint}"
        if is_development():
            best_checkpoint = abs_url(best_checkpoint)
        checkpoint_name = Path(best_checkpoint).name
        self.update_property("Best checkpoint", checkpoint_name, link=best_checkpoint)
        self.update_property(f"Best {metric_name}", f"{best_metric:.4f}", highlight=True)
        self.update_property("Task ID", str(task_id))

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "training_finished",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
            {
                "id": "evaluation_finished",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "evaluation_finished",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
            {
                "id": "new_model_better",
                "type": "source",
                "position": "right",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        return {
            "training_finished": self._process_training_finished_message,
            "evaluation_finished": self._process_evaluation_finished_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "evaluation_finished": self._send_comparison_finished_message,
            "new_model_better": self._send_new_model_better_message,
        }

    def _process_training_finished_message(self, message: TrainingFinishedMessage):
        is_first_training_message = self._last_experiment_task_id is None
        experiment = self._extract_experiment_info(message.task_id)
        self._last_experiment_task_id = message.task_id
        self._last_experiment_eval_dir = self._get_evaluation_dir_from_experiment(experiment)
        if is_first_training_message:
            # if the first training message received, store it as best experiment
            best_checkpoint = self._get_checkpoint_path(experiment)
            metric_name = experiment.get("primary_metric")
            best_metric = self._get_primary_metric_value_from_experiment(experiment)
            self._send_new_model_better_message(is_better=True, best_checkpoint=best_checkpoint)
            self._update_best_model_properties(
                best_checkpoint=best_checkpoint,
                best_metric=best_metric,
                metric_name=metric_name,
                task_id=message.task_id,
            )

    def _process_evaluation_finished_message(self, message: EvaluationFinishedMessage):
        self._best_experiment_eval_dir = message.eval_dir
        self._best_experiment_task_id = message.task_id

        if self.gui.automation_switch.is_switched():
            # assume that training message was already received
            self.run()

    def _send_comparison_finished_message(
        self,
        report_link: str,
        eval_dir: str,
    ) -> ComparisonFinishedMessage:
        return ComparisonFinishedMessage(report_link=report_link, eval_dir=eval_dir)

    def _send_new_model_better_message(
        self,
        is_better: bool,
        best_checkpoint: str,
    ) -> ComparisonFinishedMessage:
        return ComparisonFinishedMessage(is_new_best=is_better, best_checkpoint=best_checkpoint)

    # --------------------------------------------------------------------
    # Main methods -------------------------------------------------------
    # --------------------------------------------------------------------
    def run(
        self,
    ):
        """
        Sends a request to the backend to start the evaluation process.
        """
        try:
            self.gui.run_btn.disable()
            self.show_in_progress_badge("Comparison")
            if not self._last_experiment_eval_dir:
                logger.debug("Not enough evaluation directories provided for comparison.")
            elif not self._best_experiment_eval_dir:
                logger.debug(
                    "Previous best evaluation directory is not set. Using only the new one."
                )
                experiment = self._extract_experiment_info(self._last_experiment_task_id)
                best_checkpoint = self._get_checkpoint_path(experiment)
                self._best_experiment_eval_dir = self._last_experiment_eval_dir
                self._best_experiment_task_id = self._last_experiment_task_id
                self._last_experiment_eval_dir = None
                self._last_experiment_task_id = None
                self._send_new_model_better_message(is_better=True, best_checkpoint=best_checkpoint)
                metric_name = experiment.get("primary_metric")
                best_metric = self._get_primary_metric_value_from_experiment(experiment)
                self._update_best_model_properties(
                    best_checkpoint=best_checkpoint,
                    best_metric=best_metric,
                    metric_name=metric_name,
                    task_id=self._best_experiment_task_id,
                )
            else:
                eval_dirs = [self._best_experiment_eval_dir, self._last_experiment_eval_dir]
                task_info, task_status = self._start_compare_models_app(eval_dirs)
                if task_info is None:
                    raise RuntimeError("Failed to start the evaluation task.")
                task_info["evaluation_dirs"] = eval_dirs
                task_id = task_info["id"]
                task_info["status"] = task_status
                if task_status != self._api.task.Status.FINISHED:
                    self.history.add_task(task_info)
                    raise RuntimeError(f"Comparison task failed. Status: {task_status}")

                report_url = self._get_report_url(task_info)
                result_dir = self._get_evaluation_dir_from_report_url(report_url)
                # @ todo: find the best checkpoint from the evaluation results
                # self._update_automation_properties()
                task_info["result_dir"] = result_dir
                task_info["result_link"] = report_url
                self.history.add_task(task_info)

                # compare metrics to determine if the new model is better
                is_better, best_checkpoint = self._is_new_model_better(
                    self._best_experiment_task_id, self._last_experiment_task_id
                )
                if is_better:
                    self._best_experiment_eval_dir = self._last_experiment_eval_dir
                    self._best_experiment_task_id = self._last_experiment_task_id
                    self._send_new_model_better_message(
                        is_better=is_better, best_checkpoint=best_checkpoint
                    )
                    experiment = self._extract_experiment_info(self._best_experiment_task_id)
                    metric_name = experiment.get("primary_metric")
                    best_metric = self._get_primary_metric_value_from_experiment(experiment)
                    self._update_best_model_properties(
                        best_checkpoint=best_checkpoint,
                        best_metric=best_metric,
                        metric_name=metric_name,
                        task_id=self._best_experiment_task_id,
                    )
                self._send_comparison_finished_message(report_url, result_dir)
                self._last_experiment_eval_dir = None
                self._last_experiment_task_id = None

                logger.info(f"Comparison completed successfully. Task ID: {task_id}")

        except Exception as e:
            logger.error(f"Comparison failed. {repr(e)}", exc_info=True)
        finally:
            self.hide_in_progress_badge("Comparison")
            self.gui.run_btn.enable()

    def _start_compare_models_app(self, eval_dirs: List[str]) -> Tuple:
        module_id = self._api.app.get_ecosystem_module_id(self.APP_SLUG)

        logger.info("Starting Model Benchmark Evaluator task...")
        params = {"state": {"eval_dirs": eval_dirs}}
        task_info_json = self._api.task.start(
            agent_id=self.gui.agent_selector.get_value(),
            workspace_id=self.workspace_id,
            module_id=module_id,
            params=params,
        )
        task_id = task_info_json["id"]

        current_time = time.time()
        finished_statuses = [
            self._api.task.Status.FINISHED,
            self._api.task.Status.ERROR,
            self._api.task.Status.STOPPED,
            self._api.task.Status.TERMINATING,
        ]
        while (task_status := self._api.task.get_status(task_id)) not in finished_statuses:
            logger.info("Waiting for the evaluation task to start... Status: %s", task_status)
            time.sleep(5)
            if time.time() - current_time > 300:  # 5 minutes timeout
                logger.warning("Timeout reached while waiting for the evaluation task to start.")
                break

        task_info = self._api.task.get_info_by_id(task_id)
        return task_info, task_status

    # --------------------------------------------------------------------
    # Private methods ----------------------------------------------------
    # --------------------------------------------------------------------
    def _extract_experiment_info(self, task_id: int) -> Optional[float]:
        """Extract the primary metric value from the task info."""
        task_info = self._api.task.get_info_by_id(task_id)
        if not task_info:
            return None
        experiment = task_info.get("meta", {}).get("output", {}).get("experiment", {}).get("data")
        return experiment

    def _get_evaluation_dir_from_experiment(self, experiment: dict) -> Optional[str]:
        """Extract the evaluation directory from the experiment info."""
        if not experiment:
            return None
        report_id = experiment.get("evaluation_report_id")
        if report_id is None:
            return None
        file_info = self._api.file.get_info_by_id(report_id)
        if file_info is None:
            return None
        return str(Path(file_info.path).parent.parent)

    def _get_primary_metric_value_from_experiment(self, experiment: dict) -> Optional[float]:
        """Extract the primary metric value from the experiment info."""
        if not experiment:
            return None
        primary_metric = experiment.get("primary_metric")
        metrics = experiment.get("evaluation_metrics", {})
        return metrics.get(primary_metric)

    def _get_checkpoint_path(self, experiment: dict) -> Optional[str]:
        """Extract the best checkpoint path from the experiment info."""
        if not experiment:
            return None
        artifacts_dir = experiment.get("artifacts_dir")
        best_checkpoint = experiment.get("best_checkpoint")
        if artifacts_dir and best_checkpoint:
            return str(Path(artifacts_dir) / "checkpoints" / best_checkpoint)
        # checkpoint_path = f"/files/?path={checkpoint_path}"
        # if is_development():
        #     checkpoint_path = abs_url(checkpoint_path)

    def _get_report_url(self, task_info: Dict[str, Any]) -> str:
        if not task_info:
            return None
        task_output = task_info.get("meta", {}).get("output", {}).get("general", {})
        report_url = task_output.get("titleUrl")
        if report_url is None:
            return None
        if is_development():
            report_url = abs_url(report_url)
        return report_url

    def _get_evaluation_dir_from_report_url(self, report_url: str) -> Optional[str]:
        """Extract the evaluation directory from the report URL."""
        try:
            report_id = int(report_url.split("?id=")[-1])
            file_info = self._api.file.get_info_by_id(report_id)
            return str(Path(file_info.path).parent)
        except Exception:
            pass

    def _is_new_model_better(self, old_task_id: int, new_task_id: int) -> bool:
        """
        Compares the primary metrics of two checkpoints.
        Returns "better", "worse", or "equal".
        """
        old_experiment = self._extract_experiment_info(old_task_id)
        new_experiment = self._extract_experiment_info(new_task_id)

        metric_name = old_experiment.get("primary_metric")
        if metric_name != new_experiment.get("primary_metric"):
            raise ValueError("Primary metrics do not match between the two experiments.")

        old_metric = self._get_primary_metric_value_from_experiment(old_experiment)
        old_checkpoint = self._get_checkpoint_path(old_experiment)

        new_metric = self._get_primary_metric_value_from_experiment(new_experiment)
        new_checkpoint = self._get_checkpoint_path(new_experiment)
        if old_metric is None or new_metric is None:
            raise ValueError(f"Primary metric '{metric_name}' not found in evaluation results.")

        is_new_better = new_metric > old_metric
        if is_new_better:
            logger.info(f"{metric_name} of new model is better: {new_metric} > {old_metric}")
            logger.info(f"New best checkpoint path: {new_checkpoint}")
            return is_new_better, new_checkpoint

        logger.info(f"{metric_name} of new model is worse: {new_metric} <= {old_metric}")
        return is_new_better, old_checkpoint

    def save(self, enabled: Optional[bool] = None, agent_id: Optional[int] = None):
        """Save re-deploy settings."""
        if enabled is None:
            enabled = self.gui.automation_switch.is_switched()
        if agent_id is None:
            agent_id = self.gui.agent_selector.get_value()

        self.gui.save_settings(enabled, agent_id)
        self._update_automation_properties(enabled)

    def load_settings(self):
        """Load re-deploy settings from DataJson."""
        self.gui.load_settings()
        self._update_automation_properties(self.gui.automation_switch.is_switched())
