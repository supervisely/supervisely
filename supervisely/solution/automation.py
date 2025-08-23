from __future__ import annotations

from typing import Any, Callable, Optional

from supervisely.app.widgets import Button, Container, Dialog
from supervisely.solution.engine.scheduler import TasksScheduler


# ------------------------------------------------------------------
# Automation -------------------------------------------------------
# ------------------------------------------------------------------
class Automation:
    @property
    def scheduler(self):
        """Returns the scheduler for the automation."""
        return TasksScheduler()


class AutomationWidget(Automation):

    def __init__(self, func: Callable):
        """
        Initializes the automation widget.

        :param func: Function to be called when the automation is applied.
        :type func: Callable
        """
        super().__init__()
        self.func = func
        self.apply_button = Button("Apply", plain=True, button_size="small")
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id

        # --- modal ----------------------------------------------------
        self._modal: Optional[Dialog] = None
        self._open_modal_button: Optional[Button] = None

        # --- apply button ---------------------------------------------
        # should be implemented in subclasses (can not be overridden)
        # @self.apply_button.click
        # def on_apply_button_click():
        #     self.modal.hide()
        #     self.apply()

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
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
    # Automation Settings ----------------------------------------------
    # Depends on the automation GUI implementation `_create_widget()` --
    # ------------------------------------------------------------------
    def get_details(self) -> Any:
        """Returns the details of the automation."""
        raise NotImplementedError("Subclasses must implement this method")

    def save_details(self) -> None:
        """
        Saves the automation settings.

        :param enabled: Whether the automation is enabled.
        :type enabled: bool
        :param interval: Interval for synchronization.
        :type interval: int
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        :type period: str
        """
        raise NotImplementedError("Subclasses must implement this method")

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    def _create_widget(self) -> Container:
        """Create the widget for the automation."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def is_enabled(self) -> bool:
        """Implement checkbox in `_create_widget()` to check if the automation is enabled."""
        raise NotImplementedError("Subclasses must implement this method")

    # ------------------------------------------------------------------
    # Modal ------------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Returns the modal for the automation."""
        if self._modal is None:
            self._modal = Dialog(title="Automation Settings", content=self.widget)
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
