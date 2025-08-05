from typing import Tuple

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.solution.base_node import SolutionElement, SolutionCardNode

from .history import AutoImportTasksHistory


class AutoImportNode(SolutionElement):
    progress_badge_key = "Import"
    APP_SLUG = "supervisely-ecosystem/main-import"

    def __init__(
        self, api: Api, project_id: int, x: int = 0, y: int = 0, *args, **kwargs
    ):
        """
        Initialize the Manual Import GUI widget.

        :param project_id: ID of the project to import data into.
        :param widget_id: Optional widget ID for the node.
        """
        self.api = api
        self.project_id = project_id

        # --- core blocks --------------------------------------------------------
        node_id = 41  # 49
        self.tasks_history = AutoImportTasksHistory(
            self.api, self.project_id
        )  # @TODO: widget_id job_id?
        autoimport_link = abs_url(f"/import-wizard/project/{self.project_id}/dataset")
        autoimport_link += (
            f"?moduleId=435&nodeId={node_id}&appVersion=test-env&appIsBranch=true"
        )
        self.card = self._build_card(
            title="Drag & Drop Import",
            tooltip_description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time. Automatically detects 10+ annotation formats.",
            buttons=[self.tasks_history.open_modal_button],
            link=autoimport_link,
        )
        self.node = SolutionCardNode(content=self.card, x=x, y=y)

        self.modals = [self.tasks_history.modal, self.tasks_history.logs_modal]
        self._tasks = self.tasks_history._tasks
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Base Widget Methods ----------------------------------------------
    # ------------------------------------------------------------------
    def get_json_data(self) -> dict:
        """
        Returns the current data of the Manual Import widget.
        """
        return {
            "project_id": self.project_id,
            "tasks": self._tasks,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Auto Import widget.
        """
        return {}

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def update_card_after_import(self) -> Tuple[int, str]:
        """
        Updates the card after an import task is completed.

        :param task_id: Optional task ID to update the card with. If not provided, the last task will be used.
        :return: Tuple containing the number of items imported and the project image preview URL.
        """
        project = self.api.project.get_info_by_id(self.project_id)
        import_history = project.custom_data.get("import_history", {}).get("tasks", [])
        import_history_dict = {item["task_id"]: item for item in import_history}

        for task in import_history:
            if task.get("slug") == self.APP_SLUG:
                task_id = task.get("task_id")
                if task_id is not None and task_id not in self._tasks:
                    self._tasks.append(task_id)

        self.gui.card.update_property(key="Tasks", value=str(len(self._tasks)))
        if len(self._tasks) > 0:
            last_task_id = self._tasks[-1]
            last_task = import_history_dict.get(last_task_id)
            if last_task is not None:
                items_count = last_task.get("items_count", 0)
                self.gui.card.update_property(
                    key="Last import", value=f"+{items_count}"
                )
                self.gui.card.update_badge_by_key(
                    "Last import:", f"+{items_count}", "success"
                )
                return items_count, project.image_preview_url
        return None, project.image_preview_url
