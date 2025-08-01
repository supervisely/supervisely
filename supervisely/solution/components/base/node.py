from __future__ import annotations
from typing import Callable, Optional, List
from supervisely.app.widgets import SolutionCard
from supervisely.solution.base_node import SolutionElement

from .automation import BaseAutomation
from .history import BaseHistory
from .gui import BaseGUI


from typing import Callable, Dict, List, Optional

from supervisely.app import DataJson
from supervisely.app.widgets import (
    SolutionCard,
    Widget,
)
from supervisely.app.widgets_context import JinjaWidgets


class SolutionElement(Widget):
    progress_badge_key = "Task"
    gui: Optional[BaseGUI] = None
    history: Optional[BaseHistory] = None
    automation: Optional[BaseAutomation] = None

    def __new__(cls, *args, **kwargs):
        JinjaWidgets().incremental_widget_id_mode = True
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """Base class for all solution elements.

        This class is used to create a common interface for all solution elements.
        It can be extended to create specific solution elements with their own properties and methods.
        """
        widget_id = kwargs.get("widget_id", None)
        if not hasattr(self, "widget_id"):
            self.widget_id = widget_id
        Widget.__init__(self, widget_id=self.widget_id)

    # ------------------------------------------------------------------
    # Base Widget Methods ----------------------------------------------
    # ------------------------------------------------------------------
    def get_json_data(self) -> Dict:
        return {}

    def get_json_state(self) -> Dict:
        return {}

    def save_to_state(self, data: Dict) -> None:
        """Save data to the state JSON."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        if self.widget_id not in DataJson():
            DataJson()[self.widget_id] = {}
        DataJson()[self.widget_id].update(data)
        DataJson().send_changes()

    # ------------------------------------------------------------------
    # Card  ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _build_card(
        self, *, title: str, tooltip_description: str, width: int = 250
    ) -> SolutionCard:
        buttons: List = []
        if self.history is not None:
            buttons.append(self.history.open_modal_button)
        if self.automation is not None:
            buttons.append(self.automation.open_modal_button)

        return SolutionCard(
            title=title,
            tooltip=SolutionCard.Tooltip(
                description=tooltip_description,
                content=buttons,
            ),
            width=width,
        )

    # ------------------------------------------------------------------
    # Progress badge wrappers ------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_start(node: SolutionElement, func: Callable):
        def wrapped():
            node.show_in_progress_badge(node.progress_badge_key)
            if callable(func):
                func()

        return wrapped

    @staticmethod
    def _wrap_finish(node: SolutionElement, func: Callable[[int], None]):
        def wrapped(task_id: int):
            if callable(func):
                func(task_id)
            node.hide_in_progress_badge(node.progress_badge_key)

        return wrapped

    # ------------------------------------------------------------------
    # Convenience ------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, *args, **kwargs):
        if hasattr(self, "gui"):
            if hasattr(self.gui, "widget"):
                if hasattr(self.gui.widget, "run"):
                    return self.gui.widget.run(*args, **kwargs)
        raise NotImplementedError("Subclasses must implement this method")
