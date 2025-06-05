from typing import List, Optional, Tuple
from venv import logger

from supervisely._utils import abs_url
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    Checkbox,
    Container,
    Dialog,
    Empty,
    FastTable,
    Input,
    InputNumber,
    Select,
    TaskLogs,
    Text,
)
from supervisely.solution.base_node import SolutionNode
from supervisely.solution.utils import get_seconds_from_period_and_interval


class CloudImport(SolutionNode):
    """
    CloudImport node for importing data from cloud storage to a project.
    This node allows users to import data from a specified path in cloud storage
    and manage import tasks.
    """

    APP_SLUG = "e9b5a1d81aa98072cd77b402fdc122d7/cloud-storage-data-synchronizer"
    JOB_ID = "cloud_import_job"

    def __init__(
        self,
        x: int,
        y: int,
        project_id: int,
    ):
        """
        Initialize the CloudImport node.

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
        Initialize the widgets for the Cloud Import node.
        This method sets up the UI components for importing data from cloud storage.
        """
        # RUN IMPORT MODAL
        text = Text(
            "Select the folder in the Cloud Storage to import data from. Only new items will be imported.",
            status="text",
            color="gray",
        )
        self.path_input = Input(placeholder="provider://bucket-name/path/to/folder")
        self.run_btn = Button("Run", plain=True)
        run_btn_cont = Container([self.run_btn], style="align-items: flex-end")
        content = Container([text, self.path_input, run_btn_cont])

        # cloud_import_tasks = filter_tasks_by_slug(
        #     g.input_project.custom_data.get("import_history", {}).get("tasks", []),
        #     g.cloud_import_slug,
        # )

        self.run_modal = Dialog(title="Import from Cloud Storage", content=content, size="tiny")

        # AUTOMATE MODAL
        self.automate_path_input = Input(placeholder="provider://bucket-name/path/to/folder")
        self.automate_checkbox = Checkbox(content="Run every", checked=False)
        self.automate_input = InputNumber(
            min=1, value=60, debounce=1000, controls=False, size="mini"
        )
        self.automate_input.disable()
        self.automate_period_select = Select(
            [Select.Item("min", "minutes"), Select.Item("h", "hours"), Select.Item("d", "days")],
            size="mini",
        )
        self.automate_period_select.disable()
        automate_box = Container(
            [self.automate_checkbox, self.automate_input, self.automate_period_select, Empty()],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )

        self.automate_ok_btn = Button("OK", plain=True)
        automate_ok_btn_cont = Container([self.automate_ok_btn], style="align-items: flex-end")

        automate_text = Text(
            "Schedule synchronization from the Cloud Storage to the Input Project. Specify the folder path and the time interval for synchronization.",
            status="text",
            color="gray",
        )

        automation_content = Container(
            [automate_text, self.automate_path_input, automate_box, automate_ok_btn_cont]
        )
        self.automate_modal = Dialog(
            title="Automate Synchronization",
            content=automation_content,
            size="tiny",
        )

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
            columns=import_table_columns, sort_column_idx=0, fixed_columns=1, sort_order="desc"
        )
        self.logs = TaskLogs()
        self.logs_modal = Dialog(title="Task logs", content=self.logs)

        self.tasks_btn = Button(
            "Import tasks history", icon="zmdi zmdi-view-list-alt", button_size="mini"
        )
        self.tasks_modal = Dialog(title="Import tasks history", content=self.tasks_table)
        self.automate_btn = Button("Automate", icon="zmdi zmdi-flash-auto", button_size="mini")

        @self.tasks_btn.click
        def _show_tasks():
            # self.tasks_table.clear()
            for row in self._get_table_data():
                self.tasks_table.insert_row(row)
            self.tasks_modal.show()

        @self.automate_btn.click
        def automate_import():
            self.automate_modal.show()

        @self.tasks_table.row_click
        def on_row_click(clicked_row: FastTable.ClickedRow):
            self.logs.set_task_id(clicked_row.row[0])
            self.logs_modal.show()

        @self.automate_checkbox.value_changed
        def on_automate_checkbox_change(is_checked):
            if is_checked:
                self.automate_input.enable()
                self.automate_period_select.enable()
            else:
                self.automate_input.disable()
                self.automate_period_select.disable()

        tooltip = self.card_cls.Tooltip(
            description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time. Automatically detects 10+ annotation formats.",
            content=[
                self.tasks_btn,
                self.automate_btn,
            ],
        )
        self.card = self.card_cls(
            title="Import from Cloud",
            tooltip=tooltip,
            width=200,
        )

        @self.card.click
        def show_run_modal():
            self.run_modal.show()

    @property
    def modals(self) -> List[Dialog]:
        """
        Returns a list of modals used in the Cloud Import node.
        :return: List of Dialog modals.
        """
        return [
            self.run_modal,
            self.automate_modal,
            self.tasks_modal,
            self.logs_modal,
        ]

    def get_automation_details(self) -> Tuple[int, str, int, str]:
        path = self.automate_path_input.get_value()
        enabled = self.automate_checkbox.is_checked()
        period = self.automate_period_select.get_value()
        interval = self.automate_input.get_value()

        if not enabled:
            # removed = g.session.importer.unschedule_cloud_import()
            return None, None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            return None, None, None, None

        return sec, path, interval, period

    def run(self, path: str, agent_id: int) -> int:
        """
        Import data from cloud storage to input project

        :param path: Path to the folder in the Cloud Storage (e.g., "provider://bucket-name/path/to/folder")
        :type path: str
        :return: Task ID of the import task
        :rtype: int
        """
        logger.info(f"Starting import from cloud storage: {path}")

        # Get the module ID for importing from cloud
        module_id = self.api.app.get_ecosystem_module_id(self.APP_SLUG)
        module_info = self.api.app.get_ecosystem_module_info(module_id)

        # Prepare parameters for import
        params = module_info.get_arguments(images_project=self.project_id)
        params["slyFolder"] = path

        # Start import task
        session = self.api.app.start(
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=self.workspace_id,
            task_name="Import from Cloud Storage",
            params=params,
        )

        logger.info(f"Cloud import started on agent {agent_id} (task_id: {session.task_id})")
        self.append_to_data("tasks", session.task_id)

        return session.task_id

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
        :return: List of cloud import tasks ids.
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
        Returns the current data of the Cloud Import widget.
        """
        return {
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "tasks": self._tasks,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Cloud Import widget.
        """
        return {
            "automation": {
                "enabled": False,
                "path": "",
                "interval": None,
                "period": None,
            },
        }

    def save_automation_details(self, path: str, enabled: bool, interval: int, period: str) -> None:
        """
        Saves the automation details for the Cloud Import widget.
        :param path: Path to the folder in the Cloud Storage.
        :param enabled: Whether the automation is enabled.
        :param interval: Interval for synchronization.
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        """
        self.save_to_state(
            "automation",
            {"enabled": enabled, "path": path, "interval": interval, "period": period},
        )

    def update_after_import(self, task_id: Optional[int] = None) -> Tuple[int, str]:
        """Updates the card after an import task is completed.
        :param task_id: Optional task ID to update the card with. If not provided, the last task will be used.
        :return: Tuple containing the number of items imported and the project image preview URL.
        """
        tasks = self.tasks
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

    def update_automation_details(self) -> Tuple[int, str, int, str]:
        sec, path, interval, period = self.get_automation_details()
        self.automate_modal.hide()
        if path is not None:
            if self.card is not None:
                self.card.update_property(
                    "Sync", "Every {} {}".format(interval, period), highlight=True
                )
                link = abs_url(f"files/?path={path}")
                self.card.update_property("Path", path, link=link)
                logger.info(f"Added job to synchronize from Cloud Storage every {sec} sec")
                self.show_automation_badge()
        else:
            if self.card is not None:
                self.card.remove_property_by_key("Sync")
                self.card.remove_property_by_key("Path")
                # g.session.importer.unschedule_cloud_import()
                self.hide_automation_badge()

        return sec, path, interval, period
