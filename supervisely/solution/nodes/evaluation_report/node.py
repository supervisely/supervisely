import tempfile
from typing import Literal, Optional, Union

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.io.fs import silent_remove
from supervisely.io.env import team_id as env_team_id
from supervisely.sly_logger import logger
from supervisely.solution.components.link_node.node import LinkNode


class EvaluationReportNode(LinkNode):
    """
    Node for displaying a link to the Evaluation Report page.
    """

    TITLE = "Evaluation Report"
    DESCRIPTION = "Quick access to the evaluation report of the model from the Experiments. The report contains the model performance metrics and visualizations. Will be used as a reference for comparing with models from the next experiments."
    ICON = "mdi mdi-file-chart-check"
    ICON_COLOR = "#E338A7FF"
    ICON_BG_COLOR = "#FCE4F6"

    def __init__(
        self,
        width: int = 250,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        self._api = Api.from_env()
        self._team_id = env_team_id()
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        self._click_handled = True
        super().__init__(
            title=title,
            description=description,
            # link=link,
            width=width,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            tooltip_position=tooltip_position,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "evaluation_finished",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self):
        return {
            "evaluation_finished": self._process_incoming_message,
        }

    def _process_incoming_message(self, msg):
        if not hasattr(msg, "eval_dir"):
            logger.warning("Received message does not have 'eval_dir' attribute.")
            return
        eval_dir = msg.eval_dir
        self.set_report(eval_dir)

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_report(self, eval_dir: str):
        """Set the link to the evaluation report."""
        lnk_path = f"{eval_dir.rstrip('/')}/visualizations/Model Evaluation Report.lnk"
        link = self._get_url_from_lnk_path(lnk_path)
        if link:
            self.update_badge_by_key(key="Evaluation Report", label="new", badge_type="success")
            self.update_property("Evaluation Report", "open 🔗", link=link, highlight=True)
            self.set_link(link)
        else:
            self.remove_badge_by_key("Evaluation Report")
            self.remove_property_by_key("Evaluation Report")
            self.remove_link()

    def _get_url_from_lnk_path(self, remote_lnk_path) -> str:
        if not remote_lnk_path:
            logger.warning("Remote link path is empty.")
            return

        file_info = self._api.storage.get_info_by_path(self._team_id, remote_lnk_path)
        if not file_info:
            logger.warning(f"File info not found for path: {remote_lnk_path}")
            return
        temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".lnk")
        try:
            self._api.storage.download(self._team_id, remote_lnk_path, temp_file.name)
            with open(temp_file.name, "r") as f:
                line = f.readline()
                if not line.startswith("/model-benchmark?id="):
                    raise ValueError(f"Unexpected content in the file: {line}")
                report_path = line.strip()
        except Exception as e:
            logger.error(f"Failed to read the link file: {e}")
            logger.info("Trying to find the report path in Team Files...")
            report_path = remote_lnk_path.replace("Model Evaluation Report.lnk", "template.vue")
            report_info = self._api.storage.get_info_by_path(self._team_id, report_path)
            if not report_info:
                logger.error(f"Report path not found in Team Files: {report_path}")
                return
        finally:
            silent_remove(temp_file.name)
        return abs_url(report_path) if is_development() else report_path
