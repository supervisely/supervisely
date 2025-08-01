from __future__ import annotations

from typing import Optional

from supervisely.app.widgets import Button
from supervisely.solution.components.tasks_history import SolutionTasksHistory


class BaseHistory(SolutionTasksHistory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._open_modal_button: Optional[Button] = None

    @property
    def open_modal_button(self) -> Button:
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
