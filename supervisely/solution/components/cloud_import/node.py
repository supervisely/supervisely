from typing import Callable, Optional, Tuple

from supervisely.api.api import Api
from supervisely.app.widgets import SolutionCard
from supervisely.solution.base_node import SolutionCardNode
from supervisely.solution.components.base import SolutionElement

from .automation import CloudImportAutomation
from .gui import CloudImportGUI
from .history import CloudImportHistory


class CloudImportNode(SolutionElement):
    progress_badge_key = "Import"

    def __init__(self, api: Api, project_id: int, x: int = 0, y: int = 0, *args, **kwargs):
        self.api = api
        self.project_id = project_id

        # --- core blocks --------------------------------------------------------
        self.gui = CloudImportGUI(project_id=project_id)
        self.history = self.gui.task_history  # CloudImportHistory(api)
        self.automation = CloudImportAutomation(self.gui.widget.run)

        # --- card ---------------------------------------------------------------
        tooltip_desc = (
            "Each import creates a dataset folder in the Input Project, centralising all "
            "incoming data and easily managing it over time. Automatically detects 10+ annotation formats."
        )
        self.card = self._build_card(title="Import from Cloud", tooltip_description=tooltip_desc)
        # --- node ---------------------------------------------------------------
        self.node = SolutionCardNode(x=x, y=y, content=self.card)
        self.modals = [
            self.automation.modal,
            self.history.modal,
            self.history.logs_modal,
            self.gui.modal,
        ]

        @self.card.click
        def _on_card_click():
            self.gui.modal.show()

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def show_automation_badge(self, card: SolutionCard):
        sec, interval, period, path = self.automation.get_details()
        if path is not None and sec is not None:
            card.update_property("Sync", f"Every {interval} {period}", highlight=True)
            card.update_property("Path", path)
            self.node.show_automation_badge()
        else:
            card.remove_property_by_key("Sync")
            card.remove_property_by_key("Path")
            self.node.hide_automation_badge()

    def apply_automation(self, func: Optional[Callable] = None) -> None:
        self.automation.apply(func)
        self.show_automation_badge(self.card)

    # ------------------------------------------------------------------
    # Callbacks --------------------------------------------------------
    # ------------------------------------------------------------------
    def on_start(self, func: Callable[[], None]):
        return self.gui.on_start(self._wrap_start(self.node, func))

    def on_finish(self, func: Callable[[int], None]):
        return self.gui.on_finish(self._wrap_finish(self.node, func))

    # ------------------------------------------------------------------
    # Node Updates -----------------------------------------------------
    # ------------------------------------------------------------------
    def update_after_import(self, task_id: Optional[int] = None) -> Tuple[int, str]:
        tasks = self.gui.tasks
        self.card.update_property("Tasks", str(len(tasks)))
        if not tasks:
            return None, None

        task_id = task_id or tasks[-1]
        project = self.api.project.get_info_by_id(self.project_id)
        import_history = project.custom_data.get("import_history", {}).get("tasks", [])
        for imp in reversed(import_history):
            if imp.get("task_id") == task_id:
                items_count = imp.get("items_count", 0)
                self.card.update_property("Last import", f"+{items_count}")
                self.card.update_badge_by_key("Last import:", f"+{items_count}", "success")
                return items_count, project.image_preview_url
        return None, None
