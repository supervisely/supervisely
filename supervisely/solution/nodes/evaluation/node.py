import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import supervisely.io.env as env
from supervisely import logger, timeit
from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.app.widgets import Button, Dialog
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    EvaluationFinishedMessage,
    TrainingFinishedMessage,
)
from supervisely.solution.nodes.evaluation.gui import EvaluationReportGUI
from supervisely.solution.nodes.evaluation.history import EvaluationTaskHistory
from supervisely.solution.utils import find_agent, get_last_val_collection


class EvaluationNode(BaseCardNode):
    APP_SLUG = "supervisely-ecosystem/model-benchmark"
    EVALUATION_ENDPOINT = "run_evaluation"
    TITLE = "Re-evaluate on new validation dataset"
    DESCRIPTION = "Re-evaluate the best model on a new validation dataset."
    ICON = "mdi mdi-chart-box"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        project_id: Optional[int] = None,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        self._api = Api.from_env()
        project_id = project_id or env.project_id()
        self.project = self._api.project.get_info_by_id(project_id)
        self._model_path = None
        self._last_task_id = None
        self._collection_id = None
        self._model = None
        self._session_info = {}

        # --- components ---------------------------------------------------
        self.task_history = EvaluationTaskHistory()
        self.gui = EvaluationReportGUI(team_id=self.project.team_id)
        self.modals = [
            self.task_history.modal,
            self.gui.settings_modal,
            self.task_history.logs_modal,
        ]

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

        @self.gui.automation_switch.value_changed
        def on_automation_switch_change(value: bool):
            self._save(enabled=value)

        @self.gui.agent_selector.value_changed
        def on_agent_selector_change(value: int):
            self._save(agent_id=value)

        @self.click
        def show_settings():
            self.gui.settings_modal.show()

        @self.gui.run_btn.click
        def run_cb():
            self.run()

        self._update_properties(self.gui.automation_switch.is_switched())

    def _get_tooltip_buttons(self):
        self.tooltip_buttons = [self.gui.run_btn, self.task_history.btn]
        return self.tooltip_buttons

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "experiment_registered",
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
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        return {
            "experiment_registered": self._process_incomming_message,
        }

    def _available_publish_methods(self) -> Dict[str, Callable]:
        return {
            "evaluation_finished": self._send_evaluation_finished_message,
        }

    def _process_incomming_message(self, message: TrainingFinishedMessage):
        self._last_task_id = message.task_id
        self._model_path = self._extract_experiment_info(message.task_id)
        if self._model_path:
            self.run()

    def _send_evaluation_finished_message(self, res_dir: str) -> EvaluationFinishedMessage:
        return EvaluationFinishedMessage(eval_dir=res_dir, task_id=self._last_task_id)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _extract_experiment_info(self, task_id: int) -> Optional[float]:
        """Extract the primary metric value from the task info."""
        task_info = self._api.task.get_info_by_id(task_id)
        if not task_info:
            return None
        experiment = task_info.get("meta", {}).get("output", {}).get("experiment", {}).get("data")
        if not experiment:
            return None
        checkpoint = experiment.get("best_checkpoint")
        artifacts_dir = experiment.get("artifacts_dir")
        checkpoint_path = None
        if artifacts_dir and checkpoint:
            checkpoint_path = str(Path(artifacts_dir) / "checkpoints" / checkpoint)
            # checkpoint_path = f"/files/?path={checkpoint_path}"
            # if is_development():
            #     checkpoint_path = abs_url(checkpoint_path)

        return checkpoint_path

    def _save(self, enabled: Optional[bool] = None, agent_id: Optional[int] = None):
        """Save settings."""
        enabled = enabled or self.gui.automation_switch.is_switched()
        agent_id = agent_id or self.gui.agent_selector.get_value()

        self.gui.save_settings(enabled, agent_id)
        self._update_properties(enabled)

    def load_settings(self):
        """Load settings from DataJson."""
        self.gui.load_settings()
        self._update_properties(self.gui.automation_switch.is_switched())

    @timeit
    def _deploy_model(self):
        try:
            agent_id = self.gui.agent_selector.get_value()
            if not agent_id:
                agent_id = find_agent(self._api, self.project.team_id)
                if not agent_id:
                    raise ValueError("Agent ID is not set.")

            self._model = self._api.nn.deploy(
                model=self._model_path,
                workspace_id=self.project.workspace_id,
                agent_id=agent_id,
                task_name="Solution: " + str(self._api.task_id),
            )
        except TimeoutError as e:
            import re

            msg = str(e)
            match = re.search(r"Task (\d+) is not ready", msg)
            if match:
                task_id = int(match.group(1))
                self._api.task.stop(task_id)
                logger.error(f"Deployment task (id: {task_id}) timed out after 100 seconds.")
            else:
                logger.error(f"Model deployment timed out: {msg}")
            raise
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            self._model = None

    @timeit
    def _start_evaluator_session(self):
        module_id = self._api.app.get_ecosystem_module_id(self.APP_SLUG)
        task_info_json = self._api.task.start(
            agent_id=self.gui.agent_selector.get_value(),
            workspace_id=self.project.workspace_id,
            module_id=module_id,
            description=f"Evaluation started by {self._api.task_id} task",
        )
        task_id = task_info_json["id"]
        current_time = time.time()
        while self._api.task.get_status(task_id) != self._api.task.Status.STARTED:
            time.sleep(5)
            if time.time() - current_time > 300:
                break
        ready = self._api.app.wait_until_ready_for_api_calls(
            task_id=task_id, attempts=150, attempt_delay_sec=5
        )
        if not ready:
            self._api.task.stop(task_id)
            raise RuntimeError(
                f"Evaluator session (task id: {task_id}) did not start successfully after 100 seconds."
            )
        self._session_info = task_info_json

    def run(self):
        """
        Starts the evaluation process by deploying the model and starting the evaluator session.
        """
        try:
            if not self._model_path:
                logger.warning(
                    "Model path is not set. Please set the model path before running the evaluation."
                )
                return
            self.show_in_progress_badge("Evaluation")

            # create threads for deployment and evaluation sessions and start them concurrently
            deploy_thread = threading.Thread(target=self._deploy_model)
            eval_thread = threading.Thread(target=self._start_evaluator_session)
            deploy_thread.start()
            eval_thread.start()

            # wait for both threads to finish
            deploy_thread.join()
            eval_thread.join()

            # send the evaluation request in a new thread
            thread = threading.Thread(target=self._run_evaluation, daemon=True)
            thread.start()
            thread.join()

        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}", exc_info=True)
        finally:
            try:
                if self._model:
                    self._model.shutdown()
                    self._model = None
            except Exception as e:
                logger.error(f"Failed to shutdown model: {e}", exc_info=True)

            try:
                if self._session_info:
                    self._api.task.stop(self._session_info["id"])
                    self._session_info = {}
            except Exception as e:
                logger.error(f"Failed to stop evaluation session: {e}", exc_info=True)
            self.hide_in_progress_badge("Evaluation")

    def _run_evaluation(self):
        if not self._model:
            logger.error("Model is not deployed. Cannot run evaluation.")
            return
        if not self._session_info:
            logger.error("Evaluation session info is not available. Cannot run evaluation.")
            return
        collection = self._api.entities_collection.get_info_by_name(self.project.id, "main_val")
        if not collection:
            logger.error("No validation collection found. Cannot run evaluation.")
            return

        collection_id, collection_name = collection.id, collection.name
        data = {
            "session_id": self._model.task_id,
            "project_id": self.project.id,
            "collection_id": collection_id,
        }
        response = self._api.task.send_request(
            self._session_info["id"], self.EVALUATION_ENDPOINT, data=data
        )
        self._session_info["taskId"] = self._session_info["id"]
        self._session_info["sessionId"] = self._model.task_id
        self._session_info["modelPath"] = self._model_path
        self._session_info["collectionName"] = collection_name

        error = response.get("error")
        res_dir = response.get("data")
        self._session_info["status"] = "Success" if not error else "Failed"
        self.task_history.add_task(self._session_info)
        if error:
            logger.error(f"Error during evaluation: {error}")
        elif res_dir:
            self._send_evaluation_finished_message(res_dir)

    def _update_properties(self, enable: bool):
        """Update node properties with current settings."""
        value = "enabled" if enable else "disabled"
        self.update_property("Re-evaluate the best model", value, highlight=enable)
        if enable:
            self.show_automation_badge()
        else:
            self.hide_automation_badge()
            self.hide_automation_badge()
