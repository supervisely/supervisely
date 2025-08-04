from __future__ import annotations
from typing import Callable, Optional
from supervisely.app.widgets import Dialog, Widget


class BaseGUI(Widget):
    def __init__(self):
        """
        Initializes the base node GUI.
        """
        super().__init__()
        self._modal: Optional[Dialog] = None
        self.widget: Widget = self

    # ------------------------------------------------------------------
    # Properties -------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def modal(self) -> Dialog:
        """Returns the modal for the node GUI."""
        if self._modal is None:
            self._modal = Dialog(title="Node GUI", content=self.widget, size="tiny")
        return self._modal

    @property
    def tasks(self):
        """Returns the tasks list managed by the underlying widget (if any)."""
        if hasattr(self.widget, "tasks"):
            return self.widget.tasks
        raise NotImplementedError("Subclasses must implement this method")

    # ------------------------------------------------------------------
    # Optional integrated history --------------------------------------
    # ------------------------------------------------------------------
    @property
    def tasks_history(self):
        """Return an embedded TasksHistory widget if the wrapped widget exposes it (optional)."""
        return getattr(self.widget, "tasks_history", None)

    # ------------------------------------------------------------------
    # Event Handlers ---------------------------------------------------
    # ------------------------------------------------------------------
    def on_start(self, func: Callable):
        """
        Sets the function to be called when the node starts.

        :param func: Function to be called when the node starts.
        :type func: Callable
        """
        if hasattr(self.widget, "on_start"):
            return self.widget.on_start(func)
        raise NotImplementedError("Subclasses must implement this method")

    def on_finish(self, func: Callable[[int], None]):
        """
        Sets the function to be called when the node finishes.

        :param func: Function to be called when the node finishes.
        :type func: Callable[[int], None]
        """
        if hasattr(self.widget, "on_finish"):
            return self.widget.on_finish(func)
        raise NotImplementedError("Subclasses must implement this method")

    # ------------------------------------------------------------------
    # Run Node ---------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, *args, **kwargs):
        """
        Runs the node.

        :param args: Arguments to be passed to the node.
        :param kwargs: Keyword arguments to be passed to the node.
        """
        if hasattr(self.widget, "run"):
            return self.widget.run(*args, **kwargs)
        raise NotImplementedError("Subclasses must implement this method")
