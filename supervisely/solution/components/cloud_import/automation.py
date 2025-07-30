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
    Dialog,
)
from supervisely.solution.components.base import BaseAutomation
from supervisely.solution.utils import get_seconds_from_period_and_interval


class CloudImportAuto(BaseAutomation):
    def __init__(self, func: Callable):
        super().__init__()
        self.apply_btn = Button("Apply", plain=True)
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self.func = func

        # lazy UI helpers
        self._modal = None
        self._open_modal_button = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def apply(self, func: Optional[Callable] = None) -> None:
        self.func = func or self.func
        sec, path, interval, period = self.get_automation_details()
        if sec is None or path is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(self.func, sec, self.job_id, True, path)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_automation_details(self) -> Tuple[int, str, int, str]:
        path = self.path_input.get_value()
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.interval_input.get_value()

        if not enabled:
            return None, None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            return None, None, None, None

        return sec, path, interval, period

    def save_automation_details(self, path: str, enabled: bool, interval: int, period: str) -> None:
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
    # UI elements for Node integration
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Dialog that hosts the automation widget."""
        if self._modal is None:
            self._modal = Dialog(title="Automate Synchronization", content=self.widget, size="tiny")
        return self._modal

    @property
    def open_modal_button(self) -> Button:
        """Mini button that opens the automation modal."""
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
