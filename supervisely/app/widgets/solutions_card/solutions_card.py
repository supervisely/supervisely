from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Union

from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class SolutionsCard(Widget):

    class Badge:

        def __init__(
            self,
            label: str,
            on_hover: str = None,
            badge_type: Literal["info", "success", "warning", "error"] = "info",
        ):
            self._label = label
            self._on_hover = on_hover
            if badge_type not in ["info", "success", "warning", "error"]:
                raise ValueError(
                    "badge_type must be one of ['info', 'success', 'warning', 'error']"
                )
            self._badge_type = badge_type

    class Tooltip:

        def __init__(
            self,
            description: str = None,
            content: List[Widget] = None,
            properties: List = None,
        ):
            self._description = description
            self._content = content
            self._properties = properties or []

    def __init__(
        self,
        title: Optional[str] = None,
        width: Optional[Union[str, int]] = None,
        tooltip: Optional[Tooltip] = None,
        badges: List[Badge] = None,
        widget_id: Optional[str] = None,
    ):
        self._title = title
        if isinstance(width, int):
            width = f"{width}px"
        self._width = width or "100%"
        self._tooltip = tooltip
        self._show_tooltip = tooltip is not None
        self._badges = badges or []
        self._show_badges = len(self._badges) > 0
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict[str, Any]:
        return {"title": self._title, "width": self._width}

    def _prepare_badges_info(self):
        info = []
        if self._show_badges:
            for badge in self._badges[::-1]:
                badge_data = {
                    "label": badge._label,
                    "on_hover": badge._on_hover,
                    "badge_type": badge._badge_type,
                }
                info.append(badge_data)
        return info

    def get_json_state(self) -> Dict[str, Any]:
        state = {}
        state["tooltip_description"] = self._tooltip._description
        state["tooltip_properties"] = deepcopy(self._tooltip._properties)
        state["badges"] = self._prepare_badges_info()
        return state

    def update_badge(self, idx: int, label: str, on_hover: str = None, badge_type: str = None):
        if idx < 0 or idx >= len(self._badges):
            raise IndexError("Badge index out of range")

        self._badges = StateJson()[self.widget_id]["badges"]
        badge_data = {
            "label": label,
            "on_hover": self._badges[idx]["on_hover"],
            "badge_type": self._badges[idx]["badge_type"],
        }
        if on_hover is not None:
            badge_data["on_hover"] = on_hover
        if badge_type is not None:
            if badge_type not in ["info", "success", "warning", "error"]:
                raise ValueError(
                    "badge_type must be one of ['info', 'success', 'warning', 'error']"
                )
            badge_data["badge_type"] = badge_type

        StateJson()[self.widget_id]["badges"][idx] = badge_data
        self._badges = StateJson()[self.widget_id]["badges"]

        StateJson().send_changes()

    def update_property(
        self, key: str, value: str, is_link: Optional[bool] = None, highlight: Optional[bool] = None
    ):
        for prop in self._tooltip._properties:
            if prop["key"] == key:
                break
        else:
            raise KeyError(f"Property with key {key} not found")
        prop["value"] = value
        if is_link is not None:
            prop["is_link"] = is_link
        if highlight is not None:
            prop["highlight"] = highlight
        StateJson()[self.widget_id]["tooltip_properties"] = deepcopy(self._tooltip._properties)
        StateJson().send_changes()

    def add_badge(self, badge: Badge):
        StateJson()[self.widget_id]["badges"].append(
            {
                "label": badge._label,
                "on_hover": badge._on_hover,
                "badge_type": badge._badge_type,
            }
        )
        self._show_badges = True
        self._badges = StateJson()[self.widget_id]["badges"]
        StateJson().send_changes()

    def remove_badge(self, idx: int):
        if idx < 0 or idx >= len(self._badges):
            raise IndexError("Badge index out of range")

        StateJson()[self.widget_id]["badges"].pop(idx)
        if len(StateJson()[self.widget_id]["badges"]) == 0:
            self._show_badges = False
            self._badges = []
        else:
            self._badges = StateJson()[self.widget_id]["badges"]
        StateJson().send_changes()
