import threading
import time
from typing import Callable, Dict, Tuple

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.io.env import project_id as env_project_id
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import ImportFinishedMessage

from .history import AutoImportTasksHistory


class AutoImportNode(BaseCardNode):
    progress_badge_key = "Import"
    APP_SLUG = "supervisely-ecosystem/main-import"
    title = "Manual D&I Import"
    description = "Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time. Automatically detects 10+ annotation formats."
    icon = "mdi mdi-database-import"
    icon_color = "#1976D2"
    icon_bg_color = "#E3F2FD"

    def __init__(self, project_id: int = None, *args, **kwargs):
        """
        Initialize the Manual Import GUI widget.

        :param project_id: ID of the project to import data into.
        """
        self.api = Api.from_env()
        self.project_id = project_id or env_project_id()
        self._refresh_interval = 60
        self._stop_autorefresh = False
        self._refresh_thread = None
        self._last_task_id = None

        # --- core blocks --------------------------------------------------------
        self.history = AutoImportTasksHistory(self.api, self.project_id)
        self.start_autorefresh()

        # ! TODO: remove hardcoded agent_id
        agent_id = 41
        autoimport_link = abs_url(f"/import-wizard/project/{self.project_id}/dataset")
        # ! TODO: remove hardcoded agent_id
        autoimport_link += f"?moduleId=435&nodeId={agent_id}&appVersion=test-env&appIsBranch=true"

        # --- node init ----------------------------------------------------------
        title = kwargs.pop("title", self.title)
        description = kwargs.pop("description", self.description)
        icon = kwargs.pop("icon", self.icon)
        icon_color = kwargs.pop("icon_color", self.icon_color)
        icon_bg_color = kwargs.pop("icon_bg_color", self.icon_bg_color)
        self._click_handled = True
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            link=autoimport_link,
            *args,
            **kwargs,
        )

        # --- modals -------------------------------------------------------------
        self.modals = [self.history.modal, self.history.logs_modal]
        self._tasks = self.history._tasks

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "import_finished",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            }
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "import_finished": self.send_message,
        }

    def send_message(
        self,
        task_id: int,
        items_count: int,
        image_preview_url: str,
    ) -> Tuple[int, str]:
        return ImportFinishedMessage(
            task_id=task_id,
            success=True,
            items_count=items_count,
            image_preview_url=image_preview_url,
        )

    def update(self) -> Tuple[int, str]:
        """
        Updates the card after an import task is completed.

        :param task_id: Optional task ID to update the card with. If not provided, the last task will be used.
        :return: Tuple containing the number of items imported and the project image preview URL.
        """
        project = self.api.project.get_info_by_id(self.project_id)
        preview_url = project.image_preview_url

        import_history = project.custom_data.get("import_history", {}).get("tasks", [])
        import_history_dict = {item["task_id"]: item for item in import_history}

        tasks = []
        is_last_task = False
        for idx, task in enumerate(import_history[::-1]):
            if task.get("slug") == self.APP_SLUG:
                if idx == 0:
                    is_last_task = True
                task_id = task.get("task_id")
                tasks.append(task_id)

        self.update_property(key="Tasks", value=str(len(tasks)))
        if len(tasks) > 0:
            last_task_id = tasks[0]
            last_task = import_history_dict.get(last_task_id)
            if last_task is not None:
                items_count = last_task.get("items_count", 0)
                self.update_property(key="Last import", value=f"+{items_count}")
                self.update_badge_by_key("Last import", f"+{items_count}", "success")
                if is_last_task:
                    if self._last_task_id != last_task_id:
                        self._last_task_id = last_task_id
                        self.send_message(
                            task_id=last_task_id,
                            items_count=items_count,
                            image_preview_url=preview_url,
                        )
                return items_count, preview_url
        return 0, preview_url

    # ------------------------------------------------------------------
    # History Table Helpers ----------------------------------------------------
    # ------------------------------------------------------------------
    def _autorefresh(self):
        t = time.monotonic()
        while not self._stop_autorefresh:
            if time.monotonic() - t >= self._refresh_interval:
                t = time.monotonic()
                try:
                    self.update()
                except Exception as e:
                    logger.debug(f"Error during autorefresh: {e}")
            time.sleep(1)

    def stop_autorefresh(self, wait: bool = False):
        self._stop_autorefresh = True
        if wait:
            if self._refresh_thread is not None:
                self._refresh_thread.join()

    def start_autorefresh(self, interval: int = 30):
        self._refresh_interval = interval
        self._stop_autorefresh = False
        if self._refresh_thread is None:
            self._refresh_thread = threading.Thread(target=self._autorefresh, daemon=True)
        if not self._refresh_thread.is_alive():
            self._refresh_thread.start()
