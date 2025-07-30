from __future__ import annotations

from typing import Callable, Tuple, Optional, List

from supervisely.app.widgets import SolutionCard
from supervisely.solution.base_node import SolutionElement

from .automation import BaseAutomation
from .history import BaseHistory
from .gui import BaseGUI


class BaseNode(SolutionElement):
    """Base class that constructs Node and wires GUI, History and Automation helpers."""

    gui: Optional[BaseGUI] = None
    history: Optional[BaseHistory] = None
    automation: Optional[BaseAutomation] = None

    # ------------------------------------------------------------------
    # Card helpers ------------------------------------------------------
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
    # Automation helpers ------------------------------------------------
    # ------------------------------------------------------------------
    def _reflect_automation_on_card(self, card: SolutionCard):
        if self.automation is None:
            return
        sec, path, interval, period = self.automation.get_automation_details()
        if path is not None and sec is not None:
            card.update_property("Sync", f"Every {interval} {period}", highlight=True)
            card.update_property("Path", path)
            self.node.show_automation_badge()
        else:
            card.remove_property_by_key("Sync")
            card.remove_property_by_key("Path")
            self.node.hide_automation_badge()

    # ------------------------------------------------------------------
    # Run-state badge helpers ------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_start(node: SolutionElement, func: Callable):
        def wrapped():
            node.show_in_progress_badge("Run")
            if callable(func):
                func()

        return wrapped

    @staticmethod
    def _wrap_finish(node: SolutionElement, func: Callable[[int], None]):
        def wrapped(task_id: int):
            if callable(func):
                func(task_id)
            node.hide_in_progress_badge("Run")

        return wrapped
