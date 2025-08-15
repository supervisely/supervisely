from __future__ import annotations

from typing import Callable, Tuple

from supervisely.app.widgets import (Checkbox, Container, Empty, InputNumber,
                                     Select, Text)
from supervisely.solution.base_node import AutomationWidget
from supervisely.solution.utils import get_interval_period


class SmartSamplingAutomation(AutomationWidget):
    def __init__(self, func: Callable):
        super().__init__(func)

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply(self):
        enabled, _, _, sec = self.get_details()
        if not enabled:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(
                self.func, interval=sec, job_id=self.job_id, replace_existing=True
            )

    # ------------------------------------------------------------------
    # Automation Settings ----------------------------------------------
    # ------------------------------------------------------------------
    def get_details(self) -> Tuple[bool, str, int, int]:
        """
        Get the automation details from the widget.
        :return: Tuple with (enabled, period, interval, seconds)
        """
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.num_input.get_value()

        if not enabled:
            return False, None, None, None

        if period == "h":
            sec = interval * 60 * 60
        elif period == "d":
            sec = interval * 60 * 60 * 24
        else:
            sec = interval * 60
        if sec == 0:
            return False, None, None, None
        return enabled, period, interval, sec

    def save_details(self, enabled: bool, sec: int):
        """
        Saves the automation settings.

        :param enabled: Whether the automation is enabled.
        :type enabled: bool
        :param interval: Interval for synchronization.
        :type interval: int
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        :type period: str
        """
        if enabled is False:
            self.enabled_checkbox.uncheck()
        else:
            self.enabled_checkbox.check()
            period, interval = get_interval_period(sec)
            self.num_input.value = interval
            self.period_select.set_value(period)

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        """Check if the automation is enabled."""
        return self.enabled_checkbox.is_checked()

    def _create_widget(self):
        """Create the automation GUI"""
        self.enabled_checkbox = Checkbox(content="Run every", checked=False)
        self.num_input = InputNumber(min=1, value=60, debounce=1000, controls=False, size="mini")
        self.num_input.disable()
        self.period_select = Select(
            [
                Select.Item("min", "minutes"),
                Select.Item("h", "hours"),
                Select.Item("d", "days"),
            ],
            size="mini",
        )
        self.period_select.disable()
        automate_cont = Container(
            [
                self.enabled_checkbox,
                self.num_input,
                self.period_select,
                Empty(),
            ],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        apply_button_container = Container([self.apply_button], style="align-items: flex-end")
        text = Text(
            "Schedule automatic sampling data from the input project to the labeling project. <br> <strong>Note:</strong> The sampling is performed only on the images which were not copied to the labeling project yet.",
            status="text",
            color="gray",
        )

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked: bool) -> None:
            if is_checked:
                self.num_input.enable()
                self.period_select.enable()
            else:
                self.num_input.disable()
                self.period_select.disable()

        return Container([text, automate_cont, apply_button_container])
