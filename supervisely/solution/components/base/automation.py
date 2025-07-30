from __future__ import annotations

from typing import Optional

from supervisely.app.widgets import Button, Dialog
from supervisely.solution.base_node import Automation


class BaseAutomation(Automation):
    """Provides modal + opener button around Automation widget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Each subclass must define self.widget in its own __init__
        self._modal: Optional[Dialog] = None
        self._open_modal_button: Optional[Button] = None

    # ------------------------------------------------------------------
    # UI helpers --------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        if self._modal is None:
            self._modal = Dialog(title="Automation settings", content=self.widget, size="tiny")
        return self._modal

    @property
    def open_modal_button(self) -> Button:
        if self._open_modal_button is None:
            btn = Button(
                text="Automate",
                icon="zmdi zmdi-flash-auto",
                button_size="mini",
                plain=True,
                button_type="text",
            )

            @btn.click  # noqa: WPS430
            def _on_click():
                self.modal.show()

            self._open_modal_button = btn
        return self._open_modal_button
