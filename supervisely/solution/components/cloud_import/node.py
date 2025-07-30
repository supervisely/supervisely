from typing import Callable, Optional, Tuple

from supervisely._utils import abs_url, logger
from supervisely.api.api import Api
from supervisely.app.widgets import Dialog
from supervisely.solution.base_node import SolutionCardNode
from supervisely.solution.components.base import BaseNode

from .automation import CloudImportAuto
from .gui import CloudImportGUI
from .history import CloudImportHistory


class CloudImportNode(BaseNode):
    """Aggregate class that wires GUI, history and automation components together."""

    def __init__(self, api: Api, project_id: int, x: int = 0, y: int = 0, *args, **kwargs):
        self.api = api
        self.project_id = project_id

        # --- core blocks --------------------------------------------------------
        self.gui = CloudImportGUI(project_id=project_id)
        self.history = CloudImportHistory(api)
        self.automation = CloudImportAuto(self.gui.run)

        # --- UI -----------------------------------------------------------------
        tooltip_desc = (
            "Each import creates a dataset folder in the Input Project, centralising all "
            "incoming data and easily managing it over time. Automatically detects 10+ annotation formats."
        )
        self.card = self._build_card(title="Import from Cloud", tooltip_description=tooltip_desc)
        self.node = SolutionCardNode(x=x, y=y, content=self.card)
        self.run_modal = Dialog(title="Import from Cloud Storage", content=self.gui, size="tiny")

        # Modals coming from sub-components + run modal
        self.modals = [
            self.automation.modal,
            self.history.tasks_modal,
            self.history.logs_modal,
            self.run_modal,
        ]

        # Show run settings on card click
        @self.card.click
        def _on_card_click():
            self.run_modal.show()

        super().__init__(*args, **kwargs)

    # Public API helpers
    # ------------------------------------------------------------------
    def apply_automation(self, func: Optional[Callable] = None) -> None:
        """Apply (or remove) scheduling and reflect it on the card."""
        self.automation.apply(func)
        self._reflect_automation_on_card(self.card)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def on_finish(self, func: Callable[[int], None]):
        return self.gui.on_finish(self._wrap_finish(self.node, func))

    def on_start(self, func: Callable[[], None]):
        return self.gui.on_start(self._wrap_start(self.node, func))

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def update_after_import(self, task_id: Optional[int] = None) -> Tuple[int, str]:
        """Refresh card stats after import is finished."""
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
