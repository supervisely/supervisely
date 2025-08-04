from __future__ import annotations

from typing import Optional

from supervisely.app.widgets import Button, Dialog
from supervisely.solution.components.tasks_history import SolutionTasksHistory


class BaseHistory(SolutionTasksHistory):
    def __init__(self, *args, **kwargs):
        """
        Initializes the history widget.

        :param args: Arguments for the history widget.
        :param kwargs: Keyword arguments for the history widget.
        """
        super().__init__(*args, **kwargs)
        self._open_modal_button: Optional[Button] = None
        self._modal: Optional[Dialog] = None

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    # @property
    # def modal(self) -> Dialog:
    #     """Returns the modal for the history."""
    #     if self._modal is None or self.tasks_modal is None:
    #         self._modal = Dialog(title="Tasks History", content=self.widget, size="tiny")
    #     return self._modal

    @property
    def open_modal_button(self) -> Button:
        """Returns the open modal button."""
        if self._open_modal_button is None:
            btn = Button(
                text="Tasks History",
                icon="zmdi zmdi-view-list-alt",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @btn.click
            def _on_click():
                self.tasks_history.update()
                self.tasks_modal.show()

            self._open_modal_button = btn
        return self._open_modal_button
