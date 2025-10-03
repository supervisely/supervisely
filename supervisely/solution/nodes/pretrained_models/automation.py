from typing import Callable, Optional, Tuple

from supervisely.app.widgets import (
    Button,
    Checkbox,
    Container,
    Dialog,
    Empty,
    InputNumber,
    Select,
    Text,
)
from supervisely.sly_logger import logger
from supervisely.solution.components import AutomationWidget
from supervisely.solution.utils import get_interval_period, get_seconds_from_period_and_interval


class PretrainedModelsAuto(AutomationWidget):
    def __init__(self):
        self.apply_text = Text("", status="text", color="gray")
        super().__init__()

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply(self, func: Optional[Callable] = None, job_id: Optional[str] = None) -> None:
        job_id = job_id or self.job_id
        self.func = func or self.func
        enabled, _, _, sec = self.get_details()
        if not enabled:
            if self.scheduler.is_job_scheduled(job_id):
                self.scheduler.remove_job(job_id)
        else:
            self.scheduler.add_job(self.func, interval=sec, job_id=job_id, replace_existing=True)

    # ------------------------------------------------------------------
    # Automation Settings ----------------------------------------------
    # ------------------------------------------------------------------
    def get_details(self) -> Tuple[bool, str, int, int]:
        """
        Get the automation details from the widget.
        :return: Tuple with (enabled, period, interval, seconds)
        """
        enabled = self.enable_checkbox.is_checked()
        period = self.automate_period_select.get_value()
        interval = self.automate_input.get_value()

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
        if enabled:
            self.enable_checkbox.check()
        else:
            self.enable_checkbox.uncheck()

        period, interval = get_interval_period(sec)
        if period is not None and interval is not None:
            self.automate_period_select.set_value(period)
            self.automate_input.value = interval

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def enable_checkbox(self) -> Checkbox:
        if not hasattr(self, "_enable_checkbox"):
            self._enable_checkbox = Checkbox(content="Run every:", checked=False)
        return self._enable_checkbox

    def _create_widget(self):
        self.automate_input = InputNumber(
            min=1, value=60, debounce=1000, controls=False, size="mini"
        )
        self.automate_input.disable()
        self.automate_period_select = Select(
            [
                Select.Item("min", "minutes"),
                Select.Item("h", "hours"),
                Select.Item("d", "days"),
            ],
            size="mini",
        )
        self.automate_period_select.disable()
        automate_container = Container(
            [
                self.enable_checkbox,
                self.automate_input,
                self.automate_period_select,
                Empty(),
            ],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        button_container = Container([self.apply_button], style="align-items: flex-end")
        self.apply_text.set("Run training first to save settings.", "warning")

        @self.enable_checkbox.value_changed
        def on_automate_checkbox_change(is_checked: bool) -> None:
            if is_checked:
                self.enable_widgets()
            else:
                self.disable_widgets()

        return Container([self.apply_text, automate_container, button_container])

    @property
    def enable_checkbox(self) -> Checkbox:
        if not hasattr(self, "_enable_checkbox"):
            self._enable_checkbox = Checkbox(content="Run every:", checked=False)
        return self._enable_checkbox

    @property
    def is_enabled(self) -> bool:
        """Check if the automation is enabled."""
        return self.enable_checkbox.is_checked()

    def enable_widgets(self) -> None:
        """
        Enables the automation widgets for the MoveLabeled node.
        This method is called when the automation checkbox is toggled on.
        """
        self.automate_input.enable()
        self.automate_period_select.enable()

    def disable_widgets(self) -> None:
        """
        Disables the automation widgets for the MoveLabeled node.
        This method is called when the automation checkbox is toggled off.
        """
        self.automate_input.disable()
        self.automate_period_select.disable()
