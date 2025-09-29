import threading
import time
from typing import Callable, Dict, Optional, Tuple, Union

import supervisely.io.env as sly_env
from supervisely._utils import abs_url, logger
from supervisely.api.api import Api
from supervisely.app.widgets import Button
from supervisely.app.widgets import CloudImport as CloudImportWidget
from supervisely.app.widgets import Dialog, SolutionCard
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.events import publish_event
from supervisely.solution.engine.models import (
    ImportFinishedMessage,
    ImportStartedMessage,
)
from supervisely.solution.nodes.cloud_import.automation import CloudImportAutomation
from supervisely.api.task_api import TaskApi


class CloudImportNode(BaseCardNode):
    PROGRESS_BADGE_KEY = "Import"
    TITLE = "Import from Cloud"
    DESCRIPTION = "Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time. Automatically detects 10+ annotation formats."
    ICON = "mdi mdi-cloud-download"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(self, project_id: int = None, *args, **kwargs):
        """Node for importing data from the Cloud Storage to the Input Project."""
        self.api = Api.from_env()
        self.project_id = project_id or sly_env.project_id()

        # --- core blocks --------------------------------------------------------
        self.gui = CloudImportWidget(project_id=project_id)  # includes tasks history
        self.modal_content = self.gui  # for BaseCardNode
        self.history = self.gui.tasks_history
        self.automation = CloudImportAutomation()

        # --- init card ----------------------------------------------------------
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        self._click_handled = True
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        @self.click
        def show_modal():
            self.modal.show()

        @self.gui.run_btn.click
        def _on_run_btn_click():
            self.run()

        @self.automation.apply_button.click
        def _on_apply_automation_btn_click():
            self.automation.modal.hide()
            self.apply_automation(self.run)

        # --- modals -------------------------------------------------------------
        self.modals = [
            self.modal,
            self.automation.modal,
            self.gui.tasks_history.modal,
            self.gui.tasks_history.logs_modal,
        ]

    def _get_tooltip_buttons(self):
        return [self.gui.tasks_history.open_modal_button, self.automation.open_modal_button]

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
    # Modal --------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self):
        if not hasattr(self, "_modal"):
            self._modal = Dialog(
                title="Import from Cloud Storage", content=self.modal_content, size="tiny"
            )
        return self._modal

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply_automation(self, func: Optional[Callable] = None) -> None:
        self.automation.modal.hide()
        self.automation.apply(func)
        self.update_automation_details()

    def update_automation_details(self) -> Tuple[int, str, int, str]:
        sec, path, interval, period = self.automation.get_details()
        if path is not None:
            self.update_property("Sync", "Every {} {}".format(interval, period), highlight=True)
            link = abs_url(f"files/?path={path}")
            self.update_property("Path", path, link=link)
            logger.info(f"Added job to synchronize from Cloud Storage every {sec} sec")
            self.show_automation_badge()
        else:
            self.remove_property_by_key("Sync")
            self.remove_property_by_key("Path")
            self.hide_automation_badge()

        return sec, path, interval, period

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "import_finished": self.wait_import_completion,
        }

    def run(self, path: Optional[str] = None) -> int:
        """
        Runs the import task.

        :param path: Remote path to import from.
        :type path: Optional[str]
        :return: The task ID.
        :rtype: int
        """
        self.modal.hide()
        task_id = self.gui.run(path)
        self.gui.path_input.set_value("")

        # Start a separate thread to wait for the import task to complete
        self.update_badge_by_key(key="Import", label="Starting Application", badge_type="info")
        threading.Thread(target=self.handle_import_started, args=(task_id,), daemon=True).start()
        return task_id

    def handle_import_started(self, task_id: int) -> None:
        """Automatically handles import_started events"""
        # @TODO: no need?
        # self.show_in_progress_badge()
        success = self.wait_import_completion(task_id)
        logger.info(f"Import task {task_id} completed with status: {success}")
        # @TODO: no need?
        # self.hide_in_progress_badge()

    def wait_import_completion(self, task_id: int) -> ImportFinishedMessage:
        """
        Waits for the import task to complete.

        :param task_id: The ID of the import task.
        :type task_id: int
        :return: Dictionary containing the success status, items count, image preview URL, and task ID.
        :rtype: Dict[str, Optional[int]]
        """
        success = self._poll_import_progress(task_id, 2)
        items_count, image_preview_url = self.update_card_after_import(task_id)
        return ImportFinishedMessage(
            task_id=task_id,
            success=success,
            items_count=items_count,
            image_preview_url=image_preview_url,
        )

    def update_card_after_import(self, task_id: Optional[int] = None) -> Tuple[int, str]:
        """
        Updates the card after an import task is completed with the number of items imported.

        :param task_id: Optional task ID to update the card with. If not provided, the last task will be used.
        :type task_id: Optional[int]
        :return: Tuple containing the number of items imported and the project image preview URL.
        """
        tasks = self.gui.tasks
        self.update_property(key="Tasks", value=str(len(tasks)))
        if not tasks:
            return None, None

        task_id = task_id or tasks[-1]
        self.project = self.api.project.get_info_by_id(self.project_id)
        import_history = self.project.custom_data.get("import_history", {}).get("tasks", [])
        for import_task in import_history[::-1]:
            if import_task.get("task_id") == task_id:
                items_count = import_task.get("items_count", 0)
                self.update_property(key="Last import", value=f"+{items_count} images")
                self.update_badge_by_key("Last import", f"+{items_count}", "success")
                return items_count, self.project.image_preview_url
        return None, None

    # ------------------------------------------------------------------
    # Progress ---------------------------------------------------------
    # ------------------------------------------------------------------
    def _poll_import_progress(self, task_id: int, interval_sec: int = 10) -> None:
        """Poll task status every interval seconds until completion or failure."""
        while True:
            try:
                task_info = self.api.task.get_info_by_id(task_id)
            except Exception as e:
                logger.error(f"Failed to get task info for task_id={task_id}: {repr(e)}")
                return False

            if task_info is None:
                logger.error(f"Task info is not found for task_id: {task_id}")
                return False

            status = task_info.get("status")
            if status == TaskApi.Status.ERROR.value:
                self.update_badge_by_key(key="Import", label="Failed", badge_type="error")
                return False
            if status in [TaskApi.Status.STOPPED.value, TaskApi.Status.TERMINATING.value]:
                self.update_badge_by_key(key="Import", label="Stopped", badge_type="warning")
                return False
            if status == TaskApi.Status.CONSUMED.value:
                self.update_badge_by_key(key="Import", label="Consumed", badge_type="warning")
            elif status == TaskApi.Status.QUEUED.value:
                self.update_badge_by_key(key="Import", label="Queued", badge_type="warning")
            elif status == TaskApi.Status.FINISHED.value:
                self.update_badge_by_key(key="Import", label="Completed", badge_type="success")
                return True
            else:
                self._set_import_progress_from_widgets(task_id)
            time.sleep(interval_sec)

    def _set_import_progress_from_widgets(self, task_id: int) -> str:
        progress = self.api.task.get_progress(task_id)

        # Debug
        # print(f"Progress widget: {progress}")
        # with open('/root/projects/solution-labeling/task_progress.log', 'a') as f:
        #     f.write(f"Progress widget: {progress}\n\n")

        if progress is None:
            return

        name = progress["name"]
        if name in ["Uploading"]:
            label = "Uploading"
            current = progress["current"]
            total = progress["total"]
            label = f"{name}: {current} / {total}"
        else:
            return

        self.update_badge_by_key(key="Import", label=label, badge_type="info")
