from typing import Tuple

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.app.widgets import Dialog
from supervisely.solution.base_node import SolutionCardNode
from supervisely.solution.components.base import BaseNode


from .history import ManualImportHistory


class ManualImportNode(BaseNode):
    """Node that exposes Manual Drag-and-Drop import and its history."""

    APP_SLUG = "supervisely-ecosystem/main-import"

    def __init__(self, api: Api, project_id: int, x: int = 0, y: int = 0, **kwargs):
        self.api = api
        self.project_id = project_id

        # helpers -------------------------------------------------------
        self.gui = None
        self.history = ManualImportHistory(api)
        self.automation = None

        # card ----------------------------------------------------------
        tooltip = (
            "Each import creates a dataset folder in the Input Project, centralising all "
            "incoming data and easily managing it over time."
        )
        self.card = self._build_card(title="Manual Drag & Drop Import", tooltip_description=tooltip)
        self.card.link = abs_url(
            f"/import-wizard/project/{self.project_id}/dataset?moduleId=435&nodeId=41"
        )

        # node ----------------------------------------------------------
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        self.modals = [self.history.tasks_modal, self.history.logs_modal]

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Data helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def update_after_import(self) -> Tuple[int, str]:
        """Refresh card after manual import finished."""
        project = self.api.project.get_info_by_id(self.project_id)
        import_history = project.custom_data.get("import_history", {}).get("tasks", [])
        tasks_count = len(import_history)
        self.card.update_property("Tasks", str(tasks_count))

        if not import_history:
            return None, project.image_preview_url

        last = import_history[-1]
        items_count = last.get("items_count", 0)
        self.card.update_property("Last import", f"+{items_count}")
        self.card.update_badge_by_key("Last import:", f"+{items_count}", "success")
        return items_count, project.image_preview_url
