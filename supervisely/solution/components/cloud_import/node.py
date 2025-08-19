from typing import Callable, Dict, Optional, Tuple, Union

import supervisely.io.env as sly_env
from supervisely._utils import abs_url, logger
from supervisely.api.api import Api
from supervisely.app.widgets import Button
from supervisely.app.widgets import CloudImport as CloudImportWidget
from supervisely.app.widgets import Dialog, SolutionCard
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.cloud_import.automation import (
    CloudImportAutomation,
)
from supervisely.solution.engine.events import on_event, publish_event
from supervisely.solution.engine.models import (
    ImportFinishedMessage,
    ImportStartedMessage,
)


class CloudImportNode(SolutionElement):
    progress_badge_key = "Import"

    def __init__(self, project_id: int, x: int = 0, y: int = 0, *args, **kwargs):
        """Node for importing data from the Cloud Storage to the Input Project."""
        self.api = Api.from_env()
        self.project_id = project_id or sly_env.project_id()

        # --- core blocks --------------------------------------------------------
        self.gui = CloudImportWidget(project_id=project_id)  # includes tasks history
        self.automation = CloudImportAutomation(self.gui.run)
        self.card = self._build_card(
            title="Import from Cloud",
            tooltip_description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time. Automatically detects 10+ annotation formats.",
            buttons=[self.gui.tasks_history.open_modal_button, self.automation.open_modal_button],
            icon="zmdi zmdi-cloud-download",
            icon_color="#1976D2",
            icon_bg_color="#E3F2FD",
        )
        self.node = SolutionCardNode(x=x, y=y, content=self.card)

        @self.card.click
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

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self):
        if not hasattr(self, "_modal"):
            self._modal = Dialog(title="Import from Cloud Storage", content=self.gui, size="tiny")
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
            if self.card is not None:
                self.card.update_property(
                    "Sync", "Every {} {}".format(interval, period), highlight=True
                )
                link = abs_url(f"files/?path={path}")
                self.card.update_property("Path", path, link=link)
                logger.info(f"Added job to synchronize from Cloud Storage every {sec} sec")
                self._show_automation_badge()
        else:
            if self.card is not None:
                self.card.remove_property_by_key("Sync")
                self.card.remove_property_by_key("Path")
                self._hide_automation_badge()

        return sec, path, interval, period

    def _show_automation_badge(self) -> None:
        self._update_automation_badge(True)

    def _hide_automation_badge(self) -> None:
        self._update_automation_badge(False)

    def _update_automation_badge(self, enable: bool) -> None:
        for idx, prop in enumerate(self.card.badges):
            if prop["on_hover"] == "Automation":
                if enable:
                    pass  # already enabled
                else:
                    self.card.remove_badge(idx)
                return

        if enable:  # if not found
            self.card.add_badge(
                self.card.Badge(
                    label="⚡",
                    on_hover="Automation",
                    badge_type="warning",
                    plain=True,
                )
            )

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "import_started": self.run,
            "import_finished": self.wait_import_completion,
        }

    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "import_started": self.handle_import_started,
        }

    def run(self, path: Optional[str] = None) -> ImportStartedMessage:
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
        return ImportStartedMessage(task_id=task_id)

    def handle_import_started(self, message: ImportStartedMessage) -> None:
        """Automatically handles import_started events"""
        self.node.show_in_progress_badge()
        success = self.wait_import_completion(message.task_id)
        logger.info(f"Import task {message.task_id} completed with status: {success}")
        self.node.hide_in_progress_badge()

    def wait_import_completion(self, task_id: int) -> ImportFinishedMessage:
        """
        Waits for the import task to complete.

        :param task_id: The ID of the import task.
        :type task_id: int
        :return: Dictionary containing the success status, items count, image preview URL, and task ID.
        :rtype: Dict[str, Optional[int]]
        """
        success = self.gui.wait_import_completion(task_id)
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
        self.card.update_property(key="Tasks", value=str(len(tasks)))
        if not tasks:
            return None, None

        task_id = task_id or tasks[-1]
        self.project = self.api.project.get_info_by_id(self.project_id)
        import_history = self.project.custom_data.get("import_history", {}).get("tasks", [])
        for import_task in import_history[::-1]:
            if import_task.get("task_id") == task_id:
                items_count = import_task.get("items_count", 0)
                self.card.update_property(key="Last import", value=f"+{items_count}")
                self.card.update_badge_by_key("Last import:", f"+{items_count}", "success")
                return items_count, self.project.image_preview_url
        return None, None
