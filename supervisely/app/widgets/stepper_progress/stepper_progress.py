from typing import Dict, List, Literal, Optional

from supervisely.app.content import StateJson
from supervisely.app.widgets.widget import Widget


class StepperProgress(Widget):
    """A visual-only, horizontal stepper progress widget with numbered dots."""

    def __init__(
        self,
        titles: List[str],
        active_step: int = 1,
        size: Literal["small", "medium", "large"] = "medium",
        widget_id: Optional[str] = None,
    ) -> None:
        self._titles = titles or []
        self._active_step = max(1, active_step) if self._titles else 0
        allowed = ["small", "medium", "large"]
        self._size = size if size in allowed else "medium"

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        return {
            "titles": self._titles,
            "size": self._size,
        }

    def get_json_state(self) -> Dict:
        return {"active": self._active_step}

    def set_active_step(self, step: int) -> None:
        if not self._titles:
            return
        step = max(1, min(step, len(self._titles)))
        self._active_step = step
        StateJson()[self.widget_id]["active"] = self._active_step
        StateJson().send_changes()

    def next_step(self) -> None:
        self.set_active_step(self._active_step + 1)

    def previous_step(self) -> None:
        self.set_active_step(self._active_step - 1)

    def get_active_step(self) -> int:
        return int(StateJson()[self.widget_id].get("active", self._active_step))
