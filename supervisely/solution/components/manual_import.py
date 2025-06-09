from typing import List, Optional, Tuple
from venv import logger

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.app.widgets import Button, Dialog, FastTable, SolutionCard, TaskLogs
from supervisely.solution.base_node import SolutionElement, SolutionCardNode


class ManualImport(SolutionElement):
    """
    GUI for Manual Import node.
    This widget is used to create a simple drag-and-drop interface for importing data.
    """

    APP_SLUG = "supervisely-ecosystem/main-import"

    def __init__(self, api: Api, project_id: int, x: int = 0, y: int = 0):
        """
        Initialize the Manual Import GUI widget.

        :param project_id: ID of the project to import data into.
        :param widget_id: Optional widget ID for the node.
        """
        self.api = api
        self.project_id = project_id
        self.card = self._create_card()
        self.modals = [self.logs_modal, self.tasks_modal]
        self._tasks = []
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        super().__init__()

    @property
    def logs(self) -> TaskLogs:
        """
        Returns the TaskLogs widget for displaying task logs.
        """
        if not hasattr(self, "_logs"):
            self._logs = TaskLogs()
        return self._logs

    @property
    def logs_modal(self) -> Dialog:
        """
        Returns the modal dialog for displaying task logs.
        """
        if not hasattr(self, "_logs_modal"):
            self._logs_modal = Dialog(title="Task logs", content=self.logs)
        return self._logs_modal

    @property
    def tasks_table(self) -> FastTable:
        if not hasattr(self, "_tasks_table"):
            self._tasks_table = self._create_tasks_table()
        return self._tasks_table

    @property
    def tasks_modal(self) -> Dialog:
        """
        Returns the modal dialog for displaying import tasks history.
        """
        if not hasattr(self, "_tasks_modal"):
            self._tasks_modal = self._create_tasks_modal(self.tasks_table)
        return self._tasks_modal

    def _create_tasks_modal(self, tasks_table: FastTable) -> Dialog:
        """
        Creates and returns the modal dialog for displaying import tasks history.
        """
        return Dialog(title="Import tasks history", content=tasks_table)

    def _create_tasks_table(self) -> FastTable:
        """
        Creates and returns the FastTable for displaying import tasks history.
        """
        import_table_columns = [
            "Task ID",
            "App Name",
            "Dataset IDs",
            "Created At",
            "Images Count",
            "Status",
        ]
        tasks_table = FastTable(
            columns=import_table_columns,
            sort_column_idx=0,
            fixed_columns=1,
            sort_order="desc",
        )

        @tasks_table.row_click
        def on_row_click(clicked_row: FastTable.ClickedRow):
            self.logs.set_task_id(clicked_row.row[0])
            self.logs_modal.show()

        return tasks_table

    def _create_card(self) -> SolutionCard:
        """
        Creates and returns the SolutionCard for the Manual Import widget.
        """
        autoimport_link = abs_url(f"/import-wizard/project/{self.project_id}/dataset")
        autoimport_link += f"?moduleId=435&nodeId=49&appVersion=test-env&appIsBranch=true"
        return SolutionCard(
            title="Manual Drag & Drop Import",
            tooltip=self._create_tooltip(),
            width=250,
            tooltip_position="right",
            link=autoimport_link,
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        """
        Creates and returns the tooltip for the Manual Import widget.
        """
        return SolutionCard.Tooltip(
            description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time.",
            content=[self._create_tasks_button()],
        )

    def _create_tasks_button(self) -> Button:
        """
        Creates and returns the button for showing import tasks history.
        """
        btn = Button(
            "Import tasks history",
            icon="zmdi zmdi-view-list-alt",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_tasks():
            # self.tasks_table.clear()
            for row in self._get_table_data():
                self.tasks_table.insert_row(row)
            self.tasks_modal.show()

        return btn

    def get_json_data(self) -> dict:
        """
        Returns the current data of the Manual Import widget.
        """
        return {
            "project_id": self.project_id,
            "tasks": self._tasks,
        }

    def _get_table_data(self) -> List[List]:
        """
        Collects and returns the import tasks history as a list of lists.
        """

        project = self.api.project.get_info_by_id(self.project_id)
        full_history = project.custom_data.get("import_history", {}).get("tasks", [])
        history_dict = {item["task_id"]: item for item in full_history}

        for task in full_history:
            if task.get("slug") == self.APP_SLUG:
                task_id = task.get("task_id")
                if task_id is not None and task_id not in self._tasks:
                    self._tasks.append(task_id)

        data = []
        for task_id in self._tasks:
            history_item = history_dict.get(task_id)
            if history_item is None:
                data.append([task_id, "", "", "", 0, "failed"])
                continue
            if history_item.get("slug") != self.APP_SLUG:
                logger.warning(
                    f"Import history item with task_id {task_id} does not match the slug {self.APP_SLUG}. Skipping."
                )
                continue
            datasets = history_item.get("datasets", [])
            ds_ids = ", ".join(str(d["id"]) for d in datasets)
            status = history_item.get("status")
            if status == "started":
                status = "ok"
            row = [
                history_item.get("task_id"),
                history_item.get("app", {}).get("name", ""),
                ds_ids,
                history_item.get("timestamp"),
                history_item.get("items_count"),
                status,
            ]
            data.append(row)

        return data

    def update_after_import(self) -> Tuple[int, str]:
        """Updates the card after an import task is completed.
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
                self.gui.card.update_property(key="Last import", value=f"+{items_count}")
                self.gui.card.update_badge_by_key("Last import:", f"+{items_count}", "success")
                return items_count, project.image_preview_url
        return None, project.image_preview_url
