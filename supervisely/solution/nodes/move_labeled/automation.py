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


class MoveLabeledAuto(AutomationWidget):
    def __init__(self, func: Optional[Callable] = None):
        super().__init__(func)

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

    ## ------------------------------------------------------------------
    # Automation Settings ----------------------------------------------
    # ------------------------------------------------------------------
    def get_details(self) -> Tuple[bool, str, int, Optional[int], int]:
        """
        Get the automation details from the widget.
        :return: Tuple with (enabled, period, interval, seconds)
        """
        enabled = self.enable_checkbox.is_checked()
        period = self.automate_period_select.get_value()
        interval = self.automate_input.get_value()
        min_batch_enabled = self.automate_min_batch.is_checked()
        min_batch = self.automate_min_batch_input.get_value() if min_batch_enabled else None

        if not enabled:
            return None, None, None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            logger.warning("Interval must be greater than 0")
            return None, None, None, None, None

        return enabled, period, interval, min_batch, sec

    def save_details(self, enabled: bool, sec: int, min_batch: Optional[int] = None):
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
        if min_batch is not None:
            self.automate_min_batch.check()
            self.automate_min_batch_input.value = min_batch
        else:
            self.automate_min_batch.uncheck()
            self.automate_min_batch_input.value = min_batch

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
        """Create the automation GUI"""

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
        automate_box_1 = Flexbox(
            [self.enable_checkbox, self.automate_input, self.automate_period_select],
            gap=9,
            vertical_alignment="center",
        )
        self.automate_min_batch = Checkbox(content="Minimum batch size to copy", checked=False)
        self.automate_min_batch_input = InputNumber(
            min=1, value=1000, debounce=1000, controls=False, size="mini"
        )
        self.automate_min_batch.disable()
        self.automate_min_batch_input.disable()
        automate_box_2 = Container(
            [self.automate_min_batch, self.automate_min_batch_input, Empty(), Empty()],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        automation_field_1 = Field(
            automate_box_1,
            title="Enable Automation",
            description="Schedule automatic copying of labeled data (status: finished) from the labeling project to the training project.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-alarm",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )
        automation_field_2 = Field(
            automate_box_2,
            title="Minimum Batch Size",
            description="Set the minimum batch size to copy. The copying will not be performed if the number of new labeled images in the labeling project is less than the specified value.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-case-check",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        @self.enable_checkbox.value_changed
        def on_automate_checkbox_change(is_checked):
            if is_checked:
                self.enable_automation_widgets()
            else:
                self.disable_automation_widgets()

        @self.automate_min_batch.value_changed
        def on_automate_min_batch_change(is_checked):
            if is_checked:
                self.automate_min_batch_input.enable()
            else:
                self.automate_min_batch_input.disable()

        btn_cont = Container(
            [self.apply_button],
            style="align-items: flex-end",
            gap=20,
        )
        return Container([automation_field_1, automation_field_2, btn_cont])

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
