from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from supervisely.app.content import StateJson
from supervisely.app.widgets.widget import Widget


class StepperProgress(Widget):
    """A visual-only, horizontal stepper progress widget with numbered dots."""

    @dataclass
    class StepItem:
        title: str
        description: Optional[str] = None
        icon: Optional[str] = None

        def to_dict(self) -> Dict:
            return {
                "title": self.title,
                "description": self.description,
                "icon": self.icon,
            }

    def __init__(
        self,
        items: List[StepItem],
        active_step: int = 1,
        size: Literal["small", "medium", "large"] = "medium",
        simple: bool = False,
        widget_id: Optional[str] = None,
    ) -> None:
        self._items = items or []
        assert all(
            isinstance(item, StepperProgress.StepItem) for item in self._items
        ), RuntimeError("All items must be of type StepperProgress.StepItem")
        self._active_step = max(1, active_step) if self._items else 0
        size_to_spacing = {"small": 100, "medium": 150, "large": 300}
        self._size = size if size in size_to_spacing else "medium"
        self._simple = simple
        self._spacing = size_to_spacing[self._size]
        first_len = 0
        if self._items:
            first_len = len(self._items[0].title or "")
        est_half_width = (first_len * 5) / 2
        self._margin_left = int(min(max(est_half_width, 0), 180))

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "simple": self._simple,
            "align_center": True,
            "items": [item.to_dict() for item in self._items],
            "size": self._size,
        }

    def get_json_state(self) -> Dict:
        return {
            "active": self._active_step,
            "spacing": self._spacing,
            "margin_left": self._margin_left,
        }

    def set_active_step(self, step: int) -> None:
        if not self._items:
            return
        max_allowed = len(self._items) + 1
        step = max(1, min(step, max_allowed))
        self._active_step = step
        StateJson()[self.widget_id]["active"] = self._active_step
        StateJson().send_changes()

    def next_step(self) -> None:
        self.set_active_step(self._active_step + 1)

    def previous_step(self) -> None:
        self.set_active_step(self._active_step - 1)

    def get_active_step(self) -> int:
        return StateJson()[self.widget_id].get("active", self._active_step)
