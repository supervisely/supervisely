from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class SolutionCard(Widget):

    class Routes:
        CLICK = "solution_card_clicked_cb"

    class Badge:

        def __init__(
            self,
            label: str,
            on_hover: str = None,
            badge_type: Literal["info", "success", "warning", "error"] = "info",
            plain: bool = False,
        ):
            self._label = label
            self._on_hover = on_hover
            if badge_type not in ["info", "success", "warning", "error"]:
                raise ValueError(
                    "badge_type must be one of ['info', 'success', 'warning', 'error']"
                )
            self._badge_type = badge_type
            self._plain = plain

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
        tooltip: Optional[Tooltip] = None,
        content: Optional[List[Widget]] = None,
        badges: List[Badge] = None,
        tooltip_position: Literal["left", "right"] = "left",
        link: Optional[str] = None,
        width: Optional[Union[str, int]] = None,
        widget_id: Optional[str] = None,
        show_loading: Optional[bool] = True,
    ):
        self._title = title
        if isinstance(width, int):
            width = f"{width}px"
        self._width = width or "100%"
        self._tooltip = tooltip
        self._show_tooltip = tooltip is not None
        self._badges = badges or []
        self._show_badges = len(self._badges) > 0
        self._tooltip_position = tooltip_position
        self._click_handled = False
        self._show_loading = show_loading
        self._link = link
        self._content = content
        super().__init__(widget_id=widget_id, file_path=__file__)

    @property
    def badges(self) -> List[Badge]:
        self._badges = StateJson()[self.widget_id]["badges"]
        return self._badges

    @property
    def tooltip_properties(self) -> List:
        self._tooltip._properties = StateJson()[self.widget_id]["tooltip_properties"]
        return self._tooltip._properties

    @property
    def link(self) -> Optional[str]:
        return self._link

    @link.setter
    def link(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Link must be a string")
        self._link = value
        DataJson()[self.widget_id]["link"] = value
        DataJson().send_changes()

    def click(self, func: Callable[[], None]) -> Callable[[], None]:
        """Decorator that allows to handle card click. Decorated function
        will be called on card click.

        :param func: Function to be called on card click.
        :type func: Callable
        :return: Decorated function.
        :rtype: Callable
        """
        route_path = self.get_route_path(SolutionCard.Routes.CLICK)
        server = self._sly_app.get_server()
        self._click_handled = True
        DataJson()[self.widget_id]["clickable"] = True
        DataJson().send_changes()

        @server.post(route_path)
        def _click():
            if self.loading:
                return
            if self._show_loading:
                self.loading = True
            try:
                func()
            except Exception as e:
                if self._show_loading and self.loading:
                    self.loading = False
                raise e
            if self._show_loading:
                self.loading = False

        return _click

    def get_json_data(self) -> Dict[str, Any]:
        return {
            "title": self._title,
            "width": self._width,
            "clickable": self._click_handled or self._link is not None,
            "link": self._link,
        }

    def _prepare_badges_info(self):
        info = []
        if self._show_badges:
            for badge in self._badges[::-1]:
                badge_data = {
                    "label": badge._label,
                    "on_hover": badge._on_hover,
                    "badge_type": badge._badge_type,
                    "plain": badge._plain,
                }
                info.append(badge_data)
        return info

    def get_json_state(self) -> Dict[str, Any]:
        state = {}
        if self._show_tooltip:
            state["tooltip_description"] = self._tooltip._description
            state["tooltip_properties"] = deepcopy(self._tooltip._properties)
        else:
            state["tooltip_description"] = None
            state["tooltip_properties"] = []
        state["tooltip_position"] = self._tooltip_position
        state["show_tooltip"] = self._show_tooltip
        state["badges"] = self._prepare_badges_info()
        state["show_badges"] = self._show_badges
        return state

    def add_property(
        self,
        key: str,
        value: str,
        link: Optional[bool] = None,
        highlight: Optional[bool] = None,
    ):
        if link is None:
            link = False
        if highlight is None:
            highlight = False
        self._tooltip._properties.append(
            {
                "key": key,
                "value": value,
                "link": link,
                "highlight": highlight,
            }
        )
        StateJson()[self.widget_id]["tooltip_properties"] = deepcopy(self._tooltip._properties)
        StateJson().send_changes()

    def update_property(
        self, key: str, value: str, link: Optional[bool] = None, highlight: Optional[bool] = None
    ):
        for prop in self._tooltip._properties:
            if prop["key"] == key:
                break
        else:
            raise KeyError(f"Property with key {key} not found")
        prop["value"] = value
        if link is not None:
            prop["link"] = link
        if highlight is not None:
            prop["highlight"] = highlight
        StateJson()[self.widget_id]["tooltip_properties"] = deepcopy(self._tooltip._properties)
        StateJson().send_changes()

    def remove_property_by_key(self, key: str):
        found = False
        for idx, prop in enumerate(self._tooltip._properties):
            if prop["key"] == key:
                found = True
                break
        if found:
            self.remove_property(idx)

    def remove_property(self, idx: str):
        if idx < 0 or idx >= len(self._tooltip._properties):
            raise IndexError("Property index out of range")
        self._tooltip._properties.pop(idx)
        StateJson()[self.widget_id]["tooltip_properties"] = deepcopy(self._tooltip._properties)
        StateJson().send_changes()

    def add_badge(self, badge: Badge):
        StateJson()[self.widget_id]["badges"].append(
            {
                "label": badge._label,
                "on_hover": badge._on_hover,
                "badge_type": badge._badge_type,
                "plain": badge._plain,
            }
        )
        self._show_badges = True
        self._badges = StateJson()[self.widget_id]["badges"]
        StateJson()[self.widget_id]["show_badges"] = self._show_badges
        StateJson().send_changes()

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: str = None,
        plain: bool = None,
    ):
        if idx < 0 or idx >= len(self._badges):
            raise IndexError("Badge index out of range")

        self._badges = StateJson()[self.widget_id]["badges"]
        badge_data = {
            "label": label,
            "on_hover": self._badges[idx]["on_hover"],
            "badge_type": self._badges[idx]["badge_type"],
            "plain": self._badges[idx]["plain"],
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

    def remove_badge(self, idx: int):
        if idx < 0 or idx >= len(self._badges):
            raise IndexError("Badge index out of range")

        StateJson()[self.widget_id]["badges"].pop(idx)
        if len(StateJson()[self.widget_id]["badges"]) == 0:
            self._show_badges = False
            self._badges = []
            StateJson()[self.widget_id]["show_badges"] = self._show_badges
        else:
            self._badges = StateJson()[self.widget_id]["badges"]
        StateJson().send_changes()

    def remove_badge_by_key(self, key: str):
        found = False
        for idx, prop in enumerate(self._badges):
            if prop["on_hover"] == key:
                found = True
                break
        if found:
            self._badges.pop(idx)
            StateJson()[self.widget_id]["badges"] = deepcopy(self._badges)
            StateJson().send_changes()

    def set_tooltip(
        self,
        tooltip: Optional[Tooltip] = None,
        description: str = None,
        content: List[Widget] = None,
        properties: List = None,
        tooltip_position: Optional[Literal["left", "right"]] = None,
    ):
        if tooltip is not None:
            description = description or tooltip._description
            content = content or tooltip._content
            properties = properties or tooltip._properties
        if description is None and content is None and properties is None:
            raise ValueError("At least one of description, content or properties must be provided")
        self._show_tooltip = True
        tooltip = self.Tooltip(
            description=description,
            content=content,
            properties=properties,
        )
        self._tooltip = tooltip
        if tooltip_position is not None:
            self._tooltip_position = tooltip_position
            StateJson()[self.widget_id]["tooltip_position"] = tooltip_position
        StateJson()[self.widget_id]["tooltip_description"] = description
        StateJson()[self.widget_id]["tooltip_properties"] = deepcopy(properties or [])
        StateJson()[self.widget_id]["show_tooltip"] = self._show_tooltip
        StateJson().send_changes()

    def remove_tooltip(self):
        self._show_tooltip = False
        self._tooltip = None
        StateJson()[self.widget_id]["tooltip_description"] = None
        StateJson()[self.widget_id]["tooltip_properties"] = []
        StateJson()[self.widget_id]["show_tooltip"] = self._show_tooltip
        StateJson().send_changes()
