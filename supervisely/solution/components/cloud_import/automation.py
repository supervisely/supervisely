from typing import Callable, Optional

from supervisely.app.widgets import (
    Button,
    Container,
    Input,
    Text,
    Dialog,
)
from supervisely.solution.components.base import BaseAutomation


class CloudImportAutomation(BaseAutomation):
    def __init__(self, func: Callable):
        super().__init__(func)

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply(self, func: Optional[Callable] = None) -> None:
        self.func = func or self.func
        sec, interval, period, path = self.get_details()
        if sec is None or path is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(self.func, sec, self.job_id, True, path)

    # ------------------------------------------------------------------
    # Custom Automation Settings ---------------------------------------
    # ------------------------------------------------------------------
    def _create_custom_widget(self):
        description = Text(
            "Schedule synchronization from the Cloud Storage to the Input Project. Specify the folder path and the time interval for synchronization.",
            status="text",
            color="gray",
        )
        self.path_input = Input(placeholder="provider://bucket-name/path/to/folder")
        return Container([description, self.path_input])

    def get_custom_details(self) -> str:
        path = self.path_input.get_value()
        return path

    def save_custom_details(self, path: str) -> None:
        if self.path_input.get_value() != path:
            self.path_input.set_value(path)

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
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
