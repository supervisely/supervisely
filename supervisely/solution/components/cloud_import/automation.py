from typing import Callable, Optional, Tuple

from supervisely.app.widgets import (
    Button,
    Checkbox,
    Container,
    Empty,
    Input,
    InputNumber,
    Select,
    Text,
)
from supervisely.solution.base_node import AutomationWidget
from supervisely.solution.utils import get_seconds_from_period_and_interval


class CloudImportAutomation(AutomationWidget):
    """Automation settings specific to Cloud-Import synchronisation."""

    def __init__(self, func: Callable):
        super().__init__(func)

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply(self, func: Optional[Callable] = None) -> None:
        self.func = func or self.func
        sec, path, interval, period = self.get_details()
        if sec is None or path is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(self.func, sec, self.job_id, True, path)

    # ------------------------------------------------------------------
    # Automation Settings ---------------------------------------------
    # ------------------------------------------------------------------
    def get_details(self) -> Tuple[int, str, int, str]:
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

    def save_details(self, path: str, enabled: bool, interval: int, period: str) -> None:
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

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        """Check if the automation is enabled."""
        return self.enabled_checkbox.is_checked()

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

        apply_button_container = Container([self.apply_button], style="align-items: flex-end")

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked):
            if is_checked:
                self.interval_input.enable()
                self.period_select.enable()
            else:
                self.interval_input.disable()
                self.period_select.disable()

        return Container([description, self.path_input, settings_container, apply_button_container])
