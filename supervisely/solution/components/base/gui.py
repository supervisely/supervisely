from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from supervisely.app.widgets import Widget


class BaseGUI(Widget, ABC):
    """Every node-specific GUI class should inherit from this template."""

    # ------------------------------------------------------------------
    # Required API ------------------------------------------------------
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self):  # noqa: D401 – imperative; starts main action
        """Trigger the widget's primary action (import, sampling, etc.)."""

    # Optional convenience wrappers – concrete GUI may override ---------

    def on_start(self, func: Callable):  # noqa: D401 – simple passthrough
        """Connect callback executed *before* `.run()` logic starts."""
        return func

    def on_finish(self, func: Callable[[int], None]):  # noqa: D401 – simple passthrough
        """Connect callback executed after `.run()` logic ends."""
        return func

    @property
    def tasks(self):  # type: ignore[override]
        """Return list of underlying task IDs (if applicable)."""
        return []
