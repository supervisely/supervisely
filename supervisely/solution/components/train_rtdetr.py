from typing import Callable, List, Optional, Tuple
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
    SolutionCard,
    TaskLogs,
    Text,
)
from supervisely.solution.base_node import Automation, SolutionCardNode, SolutionElement
from supervisely.solution.utils import get_seconds_from_period_and_interval


class TrainRTDETRAuto(Automation):
    def __init__(self, func: Callable):
        super().__init__()
        self.apply_btn = Button("Apply", plain=True)
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self.func = func

    def apply(self, func: Optional[Callable] = None) -> None:
        self.func = func or self.func
        sec, path, interval, period = self.get_automation_details()
        if sec is None or path is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(self.func, sec, self.job_id, True, path)

    def _create_widget(self):
        description = Text(
            "Schedule RTDETR training on a regular basis.",
            status="text",
            color="gray",
        )
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

        return Container([description, settings_container, apply_btn_container])

    def get_automation_details(self) -> Tuple[int, str, int, str]:
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.interval_input.get_value()

        if not enabled:
            return None, None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            return None, None, None, None

        return sec, interval, period

    def save_automation_details(self, enabled: bool, interval: int, period: str) -> None:
        """
        Saves the automation details for the Train RTDETR node.
        :param enabled: Whether the automation is enabled.
        :param interval: Interval for synchronization.
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        """
        if self.enabled_checkbox.is_checked() != enabled:
            if enabled:
                self.enabled_checkbox.check()
            else:
                self.enabled_checkbox.uncheck()
        if self.period_select.get_value() != period:
            self.period_select.set_value(period)
        if self.interval_input.get_value() != interval:
            self.interval_input.value = interval


class TrainRTDETR(SolutionElement):
    """
    GUI for Train RTDETR node.
    """

    APP_SLUG = "supervisely-ecosystem/rt-detrv2/supervisely_integration/train"

    def __init__(self, api: Api, project_id: int, x: int = 0, y: int = 0, *args, **kwargs):
        """
        Initialize the Train RTDETR GUI widget.

        :param project_id: ID of the project to train RTDETR on.
        :param widget_id: Optional widget ID for the node.
        """
        self.api = api
        self.project_id = project_id
        self.card = self._create_card()
        self.modals = [self.logs_modal, self.tasks_modal]
        self._tasks = []
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        super().__init__(*args, **kwargs)

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
        Returns the modal dialog for displaying training tasks history.
        """
        if not hasattr(self, "_tasks_modal"):
            self._tasks_modal = self._create_tasks_modal(self.tasks_table)
        return self._tasks_modal

    def _create_tasks_modal(self, tasks_table: FastTable) -> Dialog:
        """
        Creates and returns the modal dialog for displaying training tasks history.
        """
        return Dialog(title="Training tasks history", content=tasks_table)

    def _create_tasks_table(self) -> FastTable:
        """
        Creates and returns the FastTable for displaying training tasks history.
        """
        table_columns = [
            "Task ID",
            "App Name",
            "Dataset IDs",
            "Created At",
            "Images Count",
            "Status",
        ]
        tasks_table = FastTable(
            columns=table_columns,
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
        Creates and returns the SolutionCard for the Train RTDETR node.
        """
        return SolutionCard(
            title="Train RT-DETRv2",
            tooltip=self._create_tooltip(),
            width=250,
            tooltip_position="right",
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        """
        Creates and returns the tooltip for the Train RTDETR node.
        """
        return SolutionCard.Tooltip(
            description="Train RT-DETRv2 model on your dataset.",
            content=[self._create_tasks_button()],
        )

    def _create_tasks_button(self) -> Button:
        """
        Creates and returns the button for tracking training tasks history.
        """
        btn = Button(
            "Training tasks history",
            icon="zmdi zmdi-view-list-alt",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_tasks():
            self.tasks_table.clear()
            for row in self._get_table_data():
                self.tasks_table.insert_row(row)
            self.tasks_modal.show()

        return btn

    def get_json_data(self) -> dict:
        """
        Returns the current data of the Train RTDETR node as a JSON-serializable dictionary.
        """
        return {
            "project_id": self.project_id,
            "tasks": self._tasks,
        }

    def _get_table_data(self) -> List[List]:
        """
        Collects and returns the tasks history as a list of lists.
        """

        data = []
        return data
