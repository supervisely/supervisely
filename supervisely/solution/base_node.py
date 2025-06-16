from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import SolutionCard, SolutionGraph, SolutionProject, Widget
from supervisely.solution.scheduler import TasksScheduler

T = TypeVar("T")


class SolutionElement(Widget):
    def __init__(self, *args, **kwargs):
        """Base class for all solution elements.

        This class is used to create a common interface for all solution elements.
        It can be extended to create specific solution elements with their own properties and methods.
        """
        widget_id = kwargs.get("widget_id", None)
        if not hasattr(self, "widget_id"):
            self.widget_id = widget_id
        Widget.__init__(self, widget_id=self.widget_id)

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


class Automation:
    scheduler = TasksScheduler()


# only SolutionCard/SolutionProject can be used as content
class SolutionCardNode(SolutionGraph.Node):
    def __new__(
        cls, content: Widget, x: int = 0, y: int = 0, *args, **kwargs
    ) -> SolutionGraph.Node:
        if not isinstance(content, (SolutionCard, SolutionProject)):
            raise TypeError("Content must be one of SolutionCard or SolutionProject")
        return super().__new__(cls, *args, **kwargs)

    def disable(self):
        self.content.disable()
        super().disable()

    def enable(self):
        self.content.enable()
        super().enable()

    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        for prop in self.content.tooltip_properties:
            if prop["key"] == key:
                self.content.update_property(key, value, link, highlight)
                return
        self.content.add_property(key, value, link, highlight)

    def remove_property_by_key(self, key: str):
        self.content.remove_property_by_key(key)

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: Literal["info", "success", "warning", "error"] = "info",
    ):
        self.content.update_badge(idx, label, on_hover, badge_type)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        new_key: str = None,
        plain: Optional[bool] = None,
    ):
        self.content.update_badge_by_key(
            key=key,
            label=label,
            new_key=new_key,
            badge_type=badge_type,
            plain=plain,
        )

    def add_badge(self, badge):
        self.content.add_badge(badge)

    def remove_badge(self, idx: int):
        self.content.remove_badge(idx)

    def remove_badge_by_key(self, key: str):
        self.content.remove_badge_by_key(key)

    def update_automation_badge(self, enable: bool) -> None:
        for idx, prop in enumerate(self.content.badges):
            if prop["on_hover"] == "Automation":
                if enable:
                    pass  # already enabled
                else:
                    self.content.remove_badge(idx)
                return

        if enable:  # if not found
            self.content.add_badge(
                SolutionCard.Badge(
                    label="⚡",
                    on_hover="Automation",
                    badge_type="warning",
                    plain=True,
                )
            )

    def show_automation_badge(self) -> None:
        self.update_automation_badge(True)

    def hide_automation_badge(self) -> None:
        self.update_automation_badge(False)


# only SolutionProject can be used as content
class SolutionProjectNode(SolutionCardNode):
    def __new__(
        cls, content: Widget, x: int = 0, y: int = 0, *args, **kwargs
    ) -> SolutionGraph.Node:
        if not isinstance(content, SolutionProject):
            raise TypeError("Content must be an instance of SolutionProject")
        return super().__new__(cls, content, x, y, *args, **kwargs)

    def update_preview(self, imgs: List[str], counts: List[int]):
        self.content.update_preview_url(imgs)
        self.content.update_items_count(counts)

    def update(
        self,
        project: ProjectInfo = None,
        new_items_count: int = None,
        urls: List[Union[int, str, None]] = None,
        counts: List[Union[int, None]] = None,
    ):
        if project is not None:
            self.project = project
        if new_items_count is not None:
            self.update_property(key="Last update", value=f"+{new_items_count}")
            self.update_property(key="Total", value=f"{self.project.items_count} images")
            self.update_badge_by_key(key="Last update:", label=f"+{new_items_count}")

        if self.is_training and urls is not None and counts is not None:
            self.update_preview(urls, counts)
        else:
            self.update_preview(
                [self.project.image_preview_url],
                [self.project.items_count],
            )
