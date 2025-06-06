from typing import List, Optional, Tuple
from venv import logger

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.app.widgets import Button, Dialog, FastTable, TaskLogs
from supervisely.solution.base_node import SolutionNode
from supervisely.solution.utils import get_seconds_from_period_and_interval


class ManualImport(SolutionNode):
    """
    ManualImport node for importing data using simple D&D widget.
    This node also allows users to import data from a specified path in cloud storage, agent or team files,
    and manage import tasks.
    """

    APP_SLUG = "supervisely-ecosystem/main-import"
    # JOB_ID = "manual_import_job"

    def __init__(
        self,
        x: int,
        y: int,
        project_id: int,
    ):
        """
        Initialize the ManualImport node.

        :param x: X coordinate of the node.
        :param y: Y coordinate of the node.
        :param project_id: ID of the project to import data into.
        :param widget_id: Optional widget ID for the node.
        """
        self.api = Api.from_env()
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)
        self.workspace_id = self.project.workspace_id
        self._tasks = []

        super().__init__(x, y)

    def _create_gui(self):
        """
        Initialize the widgets for the Manual Import node.
        """

        # TASKS HISTORY MODAL
        import_table_columns = [
            "Task ID",
            "App Name",
            "Dataset IDs",
            "Created At",
            "Images Count",
            "Status",
        ]
        self.tasks_table = FastTable(
            columns=import_table_columns,
            sort_column_idx=0,
            fixed_columns=1,
            sort_order="desc",
        )
        self.logs = TaskLogs()
        self.logs_modal = Dialog(title="Task logs", content=self.logs)

        self.tasks_btn = Button(
            "Import tasks history",
            icon="zmdi zmdi-view-list-alt",
            button_size="mini",
            plain=True,
            button_type="text",
        )
        self.tasks_modal = Dialog(title="Import tasks history", content=self.tasks_table)

        @self.tasks_btn.click
        def _show_tasks():
            # self.tasks_table.clear()
            for row in self._get_table_data():
                self.tasks_table.insert_row(row)
            self.tasks_modal.show()

        @self.tasks_table.row_click
        def on_row_click(clicked_row: FastTable.ClickedRow):
            self.logs.set_task_id(clicked_row.row[0])
            self.logs_modal.show()

        tooltip = self.card_cls.Tooltip(
            description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time..",
            content=[self.tasks_btn],
        )

        autoimport_link = abs_url("/import-wizard/project/{self.project_id}/dataset")
        # ! todo: remove this line ⬇︎⬇︎⬇︎⬇︎ query params only for debug
        autoimport_link += f"?moduleId=435&nodeId=49&appVersion=test-env&appIsBranch=true"

        self.card = self.card_cls(
            title="Manual Drag & Drop Import",
            tooltip=tooltip,
            width=250,
            tooltip_position="right",
            link=autoimport_link,
        )

    @property
    def modals(self) -> List[Dialog]:
        """
        Returns a list of modals used in the Manual Import node.
        :return: List of Dialog modals.
        """
        return [
            self.tasks_modal,
            self.logs_modal,
        ]

    def wait_import_completion(self, task_id: int) -> bool:
        """Wait for import task to complete and return status"""
        try:
            self.api.app.wait(task_id, target_status=self.api.task.Status.FINISHED)
            return True
        except Exception as e:
            logger.error(f"Import task {task_id} failed: {str(e)}")
            return False

    @property
    def tasks(self) -> List[int]:
        """
        Collects and returns the import tasks history.
        :return: List of import tasks ids.
        """
        self._tasks = self.data.get("tasks", [])
        return self._tasks

    def _get_table_data(self) -> List[List]:
        """
        Collects and returns the import tasks history as a list of lists.
        """

        self.project = self.api.project.get_info_by_id(self.project_id)
        full_history = self.project.custom_data.get("import_history", {}).get("tasks", [])
        history_dict = {item["task_id"]: item for item in full_history}

        data = []
        for task_id in self.tasks:
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

    def get_json_data(self) -> dict:
        """
        Returns the current data of the Manual Import widget.
        """
        return {
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "tasks": self._tasks,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Manual Import widget.
        """
        return {}

    def update_after_import(self) -> Tuple[int, str]:
        """Updates the card after an import task is completed.
        :param task_id: Optional task ID to update the card with. If not provided, the last task will be used.
        :return: Tuple containing the number of items imported and the project image preview URL.
        """
        self.project = self.api.project.get_info_by_id(self.project_id)
        import_history = self.project.custom_data.get("import_history", {}).get("tasks", [])
        import_history_dict = {item["task_id"]: item for item in import_history}

        tasks = self.tasks
        for task in import_history:
            if task.get("slug") == self.APP_SLUG:
                task_id = task.get("task_id")
                if task_id is not None and task_id not in tasks:
                    self.append_to_data("tasks", task_id)

        self.card.update_property(key="Tasks", value=str(len(self.tasks)))
        if len(self.tasks) > 0:
            last_task_id = self.tasks[-1]
            last_task = import_history_dict.get(last_task_id)
            if last_task is not None:
                items_count = last_task.get("items_count", 0)
                self.card.update_property(key="Last import", value=f"+{items_count}")
                self.card.update_badge_by_key("Last import:", f"+{items_count}", "success")
                return items_count, self.project.image_preview_url
        return None, self.project.image_preview_url
