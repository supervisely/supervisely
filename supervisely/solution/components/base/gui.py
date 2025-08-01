from __future__ import annotations
from typing import Callable, Optional
from supervisely.app.widgets import Dialog, Widget


class BaseGUI(Widget):
    def __init__(self):
        super().__init__()
        self._modal: Optional[Dialog] = None
        self.widget: Widget = self

    @property
    def modal(self) -> Dialog:
        if self._modal is None:
            self._modal = Dialog(title="Node GUI", content=self.widget, size="tiny")
        return self._modal

    @property
    def tasks(self):
        if hasattr(self.widget, "tasks"):
            return self.widget.tasks
        raise NotImplementedError("Subclasses must implement this method")

    def on_start(self, func: Callable):
        if hasattr(self.widget, "on_start"):
            return self.widget.on_start(func)
        raise NotImplementedError("Subclasses must implement this method")

    def on_finish(self, func: Callable[[int], None]):
        if hasattr(self.widget, "on_finish"):
            return self.widget.on_finish(func)
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, *args, **kwargs):
        if hasattr(self.widget, "run"):
            return self.widget.run(*args, **kwargs)
        raise NotImplementedError("Subclasses must implement this method")
