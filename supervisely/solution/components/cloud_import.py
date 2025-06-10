from typing import Callable, Optional, Tuple

from supervisely._utils import abs_url, logger
from supervisely.api.api import Api
from supervisely.app.widgets import CloudImport as CloudImportWidget
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
    SolutionCard,
    TaskLogs,
    Text,
)
from supervisely.solution.base_node import Automation, SolutionElement, SolutionCardNode
from supervisely.solution.utils import get_seconds_from_period_and_interval


class CloudImportAuto(Automation):
    def __init__(self, func: Callable):
        super().__init__()
        self.apply_btn = Button("Apply", plain=True)
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self.func = func

    def apply(self):
        sec, path, interval, period = self.get_automation_details()
        if sec is None or path is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(
                self.func, interval=sec, job_id=self.job_id, replace_existing=True
            )

    def _create_widget(self):
        description = Text(
            "Schedule synchronization from the Cloud Storage to the Input Project. Specify the folder path and the time interval for synchronization.",
            status="text",
            color="gray",
        )
        self.path_input = Input(placeholder="provider://bucket-name/path/to/folder")
        self.enabled_checkbox = Checkbox(content="Run every", checked=False)
        self.interval_input = InputNumber(
            min=1, value=60, debounce=1000, controls=False, size="mini"
        )
        self.interval_input.disable()
        self.period_select = Select(
            [Select.Item("min", "minutes"), Select.Item("h", "hours"), Select.Item("d", "days")],
            size="mini",
        )
        self.period_select.disable()

        settings_container = Container(
            [self.enabled_checkbox, self.interval_input, self.period_select, Empty()],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )

        apply_btn_container = Container([self.apply_btn], style="align-items: flex-end")

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked):
            if is_checked:
                self.interval_input.enable()
                self.period_select.enable()
            else:
                self.interval_input.disable()
                self.period_select.disable()

        return Container([description, self.path_input, settings_container, apply_btn_container])

    def get_automation_details(self) -> Tuple[int, str, int, str]:
        path = self.path_input.get_value()
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.interval_input.get_value()

        if not enabled:
            # removed = g.session.importer.unschedule_cloud_import()
            return None, None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            return None, None, None, None

        return sec, path, interval, period

    def save_automation_details(self, path: str, enabled: bool, interval: int, period: str) -> None:
        """
        Saves the automation details for the Cloud Import widget.
        :param path: Path to the folder in the Cloud Storage.
        :param enabled: Whether the automation is enabled.
        :param interval: Interval for synchronization.
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        """
        if self.path_input.get_value() != path:
            self.path_input.set_value(path)
        if self.enabled_checkbox.is_checked() != enabled:
            if enabled:
                self.enabled_checkbox.check()
            else:
                self.enabled_checkbox.uncheck()
        if self.period_select.get_value() != period:
            self.period_select.set_value(period)
        if self.interval_input.get_value() != interval:
            self.interval_input.value = interval


class CloudImport(SolutionElement):
    def __init__(self, api: Api, project_id: int, x: int = 0, y: int = 0):
        self.api = api
        self.project_id = project_id
        self.main_widget = CloudImportWidget(project_id=project_id)
        self.automation = CloudImportAuto(self.main_widget.run)
        self.card = self._create_card()
        self.sync_modal = self._create_automation_modal()
        w, h = (
            None,
            None,
        )  # TODO: width and height are properties of the SolutionCard, but not used in the original code
        self.node = SolutionCardNode(x=x, y=y, content=self.card)

        @self.card.click
        def show_run_modal():
            self.run_modal.show()

        self.modals = [self.tasks_modal, self.sync_modal, self.logs_modal, self.run_modal]

        super().__init__()

    @property
    def logs(self):
        if not hasattr(self, "_logs"):
            self._logs = TaskLogs()
        return self._logs

    @property
    def logs_modal(self):
        if not hasattr(self, "_logs_modal"):
            self._logs_modal = Dialog(title="Task logs", content=self.logs)
        return self._logs_modal

    @property
    def tasks_table(self):
        if not hasattr(self, "_tasks_table"):
            self._tasks_table = self._create_tasks_history_table()

            @self._tasks_table.row_click
            def on_row_click(clicked_row: FastTable.ClickedRow):
                self.logs.set_task_id(clicked_row.row[0])
                self.logs_modal.show()

        return self._tasks_table

    @property
    def tasks_modal(self):
        if not hasattr(self, "_tasks_modal"):
            self._tasks_modal = self._create_tasks_modal(self.tasks_table)
        return self._tasks_modal

    @property
    def run_modal(self):
        if not hasattr(self, "_run_modal"):
            self._run_modal = Dialog(
                title="Import from Cloud Storage", content=self.main_widget, size="tiny"
            )
        return self._run_modal

    def _create_tasks_modal(self, tasks_table: FastTable):
        return Dialog(title="Import tasks history", content=tasks_table)

    def _create_tasks_history_table(self):
        import_table_columns = [
            "Task ID",
            "App Name",
            "Dataset IDs",
            "Created At",
            "Images Count",
            "Status",
        ]
        return FastTable(
            columns=import_table_columns, sort_column_idx=0, fixed_columns=1, sort_order="desc"
        )

    def _create_tasks_button(self):
        btn = Button(
            "Import tasks history",
            icon="zmdi zmdi-view-list-alt",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_tasks_dialog():
            for row in self.main_widget._get_table_data():
                self.tasks_table.insert_row(row)
            self.tasks_modal.show()

        return btn

    def _create_automation_modal(self):
        self.automation.apply_btn.click(self.update_automation_details())

        return Dialog(
            title="Automate Synchronization",
            content=self.automation.widget,
            size="tiny",
        )

    def _create_automation_button(self):
        btn = Button(
            "Automate",
            icon="zmdi zmdi-flash-auto",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_automate_dialog():
            self.sync_modal.show()

        return btn

    def _create_tooltip(self):
        return SolutionCard.Tooltip(
            description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time. Automatically detects 10+ annotation formats.",
            content=[
                self._create_tasks_button(),
                self._create_automation_button(),
            ],
        )

    def _create_card(self) -> SolutionCard:
        return SolutionCard(
            title="Import from Cloud",
            tooltip=self._create_tooltip(),
            width=250,
        )

    def update_after_import(self, task_id: Optional[int] = None) -> Tuple[int, str]:
        """Updates the card after an import task is completed.
        :param task_id: Optional task ID to update the card with. If not provided, the last task will be used.
        :return: Tuple containing the number of items imported and the project image preview URL.
        """
        tasks = self.main_widget.tasks
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

    def apply_automation(self) -> None:
        self.automation.apply()
        self.update_automation_details()

    def update_automation_details(self) -> Tuple[int, str, int, str]:
        sec, path, interval, period = self.automation.get_automation_details()
        # self.sync_modal.hide()
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
                # g.session.importer.unschedule_cloud_import()
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
