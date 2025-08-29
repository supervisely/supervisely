from typing import Callable, Optional, Tuple

from supervisely.app.widgets import (
    Button,
    Checkbox,
    Flexbox,
    Container,
    Dialog,
    Empty,
    Field,
    InputNumber,
    Select,
    Switch,
    Text,
)
from supervisely.sly_logger import logger
from supervisely.solution.automation import Automation, AutomationWidget
from supervisely.solution.utils import get_interval_period, get_seconds_from_period_and_interval


class PretrainedModelsAuto(AutomationWidget):
    def __init__(self):
        self.apply_btn = Button("Apply", plain=True)
        self.apply_text = Text("", status="text", color="gray")
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self.func = None
        super().__init__()

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply(self, func: Optional[Callable] = None) -> None:
        self.func = func or self.func
        enabled, _, _, _, sec = self.get_details()
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
    def get_details(self) -> Tuple[bool, str, int, Optional[int], int]:
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

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Returns the modal for the automation."""
        if self._modal is None:
            self._modal = Dialog(title="Automation Settings", content=self.widget, size="tiny")
        return self._modal

    def _create_widget(self):
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
        apply_btn = Container([self.apply_btn], style="align-items: flex-end")
        self.apply_text.set("Run training first to save settings.", "warning")

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked: bool) -> None:
            if is_checked:
                self.num_input.enable()
                self.period_select.enable()
            else:
                self.num_input.disable()
                self.period_select.disable()

        return Container([self.apply_text, automate_cont, apply_btn])

    @property
    def enable_checkbox(self) -> Checkbox:
        if not hasattr(self, "_enable_checkbox"):
            self._enable_checkbox = Checkbox(content="Run every:", checked=False)
        return self._enable_checkbox

    @property
    def is_enabled(self) -> bool:
        """Check if the automation is enabled."""
        return self.enable_checkbox.is_checked()

    def enable_automation_widgets(self) -> None:
        """
        Enables the automation widgets for the MoveLabeled node.
        This method is called when the automation checkbox is toggled on.
        """
        self.automate_input.enable()
        self.automate_period_select.enable()
        self.automate_min_batch.enable()

    def disable_automation_widgets(self) -> None:
        """
        Disables the automation widgets for the MoveLabeled node.
        This method is called when the automation checkbox is toggled off.
        """
        self.automate_input.disable()
        self.automate_period_select.disable()
        self.automate_min_batch.uncheck()
        self.automate_min_batch.disable()
        self.automate_min_batch_input.disable()
        self.automate_min_batch.uncheck()
        self.automate_min_batch.disable()
        self.automate_min_batch_input.disable()
