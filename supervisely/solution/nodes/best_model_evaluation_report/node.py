from supervisely.sly_logger import logger
from supervisely.solution.engine.models import EvaluationFinishedMessage
from supervisely.solution.nodes.evaluation_report.node import EvaluationReportNode


class BestModelEvaluationReportNode(EvaluationReportNode):
    """
    Node for displaying a link to the Evaluation Report page of the best model.
    """

    TITLE = "Evaluation Report"
    DESCRIPTION = "Quick access to the evaluation report of the best model from the Experiments. The report contains the model performance metrics and visualizations. Will be used as a reference for comparing with models from the next experiments."
    ICON = "mdi mdi-file-chart-check"
    ICON_COLOR = "#E338A7FF"
    ICON_BG_COLOR = "#FCE4F6"

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
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
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self):
        return super()._available_subscribe_methods()

    def _available_publish_methods(self):
        return {
            "evaluation_finished": self._send_evaluation_finished_message,
        }

    def _process_incoming_message(self, msg):
        if not hasattr(msg, "eval_dir"):
            logger.warning("Received message does not have 'eval_dir' attribute.")
            return
        eval_dir = msg.eval_dir
        self.set_report(eval_dir)
        self._send_evaluation_finished_message(eval_dir=eval_dir, task_id=msg.task_id)

    def _send_evaluation_finished_message(
        self, eval_dir: str, task_id: int
    ) -> EvaluationFinishedMessage:
        return EvaluationFinishedMessage(eval_dir=eval_dir, task_id=task_id)
