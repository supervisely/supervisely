from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget


class SolutionProject(Widget):

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
        preview_url: Union[Optional[str], List[str]] = None,
        project_id: Optional[int] = None,
        items_count: Union[Optional[int], List[int]] = None,
        items_type: Optional[str] = "images",
        width: Optional[Union[str, int]] = None,
        tooltip: Optional[Tooltip] = None,
        badges: List[Badge] = None,
        tooltip_position: Literal["left", "right"] = "left",
        widget_id: Optional[str] = None,
    ):
        self._api = Api.from_env()
        if isinstance(items_count, int):
            items_count = [items_count]
        self._items_count = items_count
        self._items_type = items_type
        self._preview_url = preview_url
        # if project_id is None and preview_url is None:
        #     raise ValueError("Either project_id or preview_url must be provided")
        if preview_url is not None:
            if isinstance(preview_url, str):
                preview_url = [preview_url]
            self._preview_url = preview_url
        if project_id is not None and self._preview_url is None:
            project = self._api.project.get_info_by_id(project_id)
            self._preview_url = [project.image_preview_url]
            self._items_count = [project.items_count]

        if self._items_count is not None:
            self._items_count = [f"{count} {self._items_type}" for count in self._items_count]

            if len(self._items_count) != len(self._preview_url):
                raise ValueError(
                    f"Length of preview_url ({len(self._preview_url)}) must be equal to length of items_count ({len(self._items_count)})"
                )

        self._title = title
        if isinstance(width, int):
            width = f"{width}px"

        if self._items_count is not None and len(self._items_count) > 1 and width is None:
            width = "270px"

        self._width = width or "100%"
        self._tooltip = tooltip
        self._show_tooltip = tooltip is not None
        self._badges = badges or []
        self._show_badges = len(self._badges) > 0
        self._tooltip_position = tooltip_position

        # not implemented yet
        self._click_handled = False
        self._show_loading = False
        self._link = None
        self._content = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    @property
    def badges(self) -> List[Badge]:
        self._badges = StateJson()[self.widget_id]["badges"]
        return self._badges

    @property
    def tooltip_properties(self) -> List:
        self._tooltip._properties = StateJson()[self.widget_id]["tooltip_properties"]
        return self._tooltip._properties

    def get_json_data(self) -> Dict[str, Any]:
        return {
            "title": self._title,
            "width": self._width,
            "preview_urls": self._preview_url,
            "items_counts": self._items_count,
            "clickable": False,
        }

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
        if self._show_tooltip:
            state["tooltip_description"] = self._tooltip._description
            state["tooltip_properties"] = deepcopy(self._tooltip._properties)
        else:
            state["tooltip_description"] = None
            state["tooltip_properties"] = []
        state["badges"] = self._prepare_badges_info()
        state["show_badges"] = self._show_badges
        return state

    def update_preview_url(self, preview_url: Union[str, List[str]]):
        if isinstance(preview_url, str):
            preview_url = [preview_url]
        self._preview_url = preview_url
        DataJson()[self.widget_id]["preview_urls"] = self._preview_url
        DataJson().send_changes()

    def update_items_count(self, items_count: Union[int, str, List[Union[int, str]]]):
        if isinstance(items_count, int):
            items_count = [f"{items_count} {self._items_type}"]
        if isinstance(items_count, str):
            items_count = [items_count]
        if isinstance(items_count, list):
            for i in range(len(items_count)):
                if isinstance(items_count[i], int):
                    items_count[i] = f"{items_count[i]} {self._items_type}"
                # elif not isinstance(items_count[i], str):
                #     raise TypeError(
                #         f"items_count[{i}] must be either int or str, but got {type(items_count[i])}"
                #     )

        self._items_count = items_count
        DataJson()[self.widget_id]["items_counts"] = self._items_count
        DataJson().send_changes()

    def add_property(
        self,
        key: str,
        value: str,
        is_link: Optional[bool] = None,
        highlight: Optional[bool] = None,
    ):
        if is_link is None:
            is_link = False
        if highlight is None:
            highlight = False
        self._tooltip._properties.append(
            {
                "key": key,
                "value": value,
                "is_link": is_link,
                "highlight": highlight,
            }
        )
        StateJson()[self.widget_id]["tooltip_properties"] = deepcopy(self._tooltip._properties)
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
        StateJson()[self.widget_id]["show_badges"] = self._show_badges
        self._badges = StateJson()[self.widget_id]["badges"]
        StateJson().send_changes()

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

    def set_project(self, project: ProjectInfo) -> None:
        self._preview_url = [project.image_preview_url]
        self._items_count = [f"{project.items_count or 0} {self._items_type}"]
        DataJson()[self.widget_id]["preview_urls"] = self._preview_url
        DataJson()[self.widget_id]["items_counts"] = self._items_count
        DataJson().send_changes()
