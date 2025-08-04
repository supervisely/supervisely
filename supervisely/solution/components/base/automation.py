from __future__ import annotations
from typing import Callable, Optional, Tuple
from supervisely.solution.scheduler import TasksScheduler
from supervisely.solution.utils import get_seconds_from_period_and_interval
from supervisely.app.widgets import (
    Button,
    Checkbox,
    Container,
    Empty,
    InputNumber,
    Select,
    Text,
    Dialog,
)


class Automation:
    @property
    def scheduler(self):
        """Returns the scheduler for the automation."""
        return TasksScheduler()


class AutomationWidget(Automation):

    def __init__(self, description: str, func: Callable):
        """
        Initializes the automation widget.

        :param description: Description of the automation.
        :type description: str
        :param func: Function to be called when the automation is applied.
        :type func: Callable
        """
        super().__init__()
        self.description = description
        self.func = func
        self.apply_button = Button("Apply", plain=True, button_size="small")
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id

        self._modal: Optional[Dialog] = None
        self._open_modal_button: Optional[Button] = None

        @self.apply_button.click
        def on_apply_button_click():
            self.apply()

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        """Check if the automation is enabled."""
        return self.enabled_checkbox.is_checked()

    def apply(self) -> None:
        """Applies the automation settings."""
        sec, _, _ = self.get_details()
        if sec is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(self.func, sec, self.job_id, True)
        if self._on_apply is not None:
            self._on_apply()

    def on_apply(self, func: Callable) -> None:
        """Sets the function to be called when the automation is applied."""
        self._on_apply = func

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    def _create_widget(self) -> Container:
        """Creates the widget for the automation."""
        self.description = Text(
            self.description,
            status="text",
            color="gray",
        )
        self.enabled_checkbox = Checkbox(content="Run every", checked=False)
        self.interval_input = InputNumber(
            min=1, value=60, debounce=1000, controls=False, size="mini", width=150
        )
        self.interval_input.disable()
        self.period_select = Select(
            [Select.Item("min", "minutes"), Select.Item("h", "hours"), Select.Item("d", "days")],
            size="mini",
        )
        self.period_select.disable()

        settings_container = Container(
            [
                self.enabled_checkbox,
                self.interval_input,
                self.period_select,
                Empty(),
            ],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
            overflow="wrap",
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

        return Container([self.description, settings_container, apply_button_container])

    @property
    def modal(self) -> Dialog:
        """Returns the modal for the automation."""
        if self._modal is None:
            self._modal = Dialog(title="Automation Settings", content=self.widget, size="tiny")
        return self._modal

    @property
    def open_modal_button(self) -> Button:
        """Returns the open modal button."""
        if self._open_modal_button is None:
            btn = Button(
                text="Automate",
                icon="zmdi zmdi-flash-auto",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @btn.click
            def _on_click():
                self.modal.show()

            self._open_modal_button = btn
        return self._open_modal_button

    # ------------------------------------------------------------------
    # Automation Settings ----------------------------------------------
    # ------------------------------------------------------------------
    def get_details(self) -> Tuple[int, int, str]:
        """Returns the details of the automation."""
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.interval_input.get_value()

        if not enabled:
            return None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            return None, None, None

        return sec, interval, period

    def save_details(self, enabled: bool, interval: int, period: str) -> None:
        """
        Saves the automation settings.

        :param enabled: Whether the automation is enabled.
        :type enabled: bool
        :param interval: Interval for synchronization.
        :type interval: int
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        :type period: str
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
