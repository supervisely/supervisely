from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

from supervisely.api.project_api import ProjectInfo
from supervisely.app import DataJson
from supervisely.app.widgets import (
    Button,
    Checkbox,
    Container,
    Empty,
    InputNumber,
    Select,
    SolutionCard,
    SolutionGraph,
    SolutionProject,
    Text,
    Widget,
)
from supervisely.app.widgets_context import JinjaWidgets
from supervisely.solution.scheduler import TasksScheduler
from supervisely.solution.utils import get_seconds_from_period_and_interval


class SolutionElement(Widget):

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
    @property
    def scheduler(self):
        return TasksScheduler()


class AutomationWidget(Automation):

    def __init__(self, description: str, func: Callable):
        super().__init__()
        self.description = description
        self.func = func
        self.apply_btn = Button("Apply", plain=True, button_size="small")
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self._on_apply = None

        @self.apply_btn.click
        def on_apply_btn_click():
            self.apply()

    def apply(self) -> None:
        sec, _, _ = self.get_automation_details()
        if sec is None:
            if self.scheduler.is_job_scheduled(self.job_id):
                self.scheduler.remove_job(self.job_id)
        else:
            self.scheduler.add_job(self.func, sec, self.job_id, True)
        if self._on_apply is not None:
            self._on_apply()

    def on_apply(self, func: Callable) -> None:
        self._on_apply = func

    def is_enabled(self) -> bool:
        """Check if the automation is enabled."""
        return self.enabled_checkbox.is_checked()

    def _create_widget(self):
        self.description = Text(
            self.description,
            status="text",
            color="gray",
        )
        self.enabled_checkbox = Checkbox(content="Run every", checked=False)
        self.interval_input = InputNumber(
            min=1, value=60, debounce=1000, controls=False, size="mini", width=150
        )
        self.interval_input.disable()
        self.period_select = Select(
            [Select.Item("min", "minutes"), Select.Item("h", "hours"), Select.Item("d", "days")],
            size="mini",
        )
        self.period_select.disable()

        settings_container = Container(
            [
                self.enabled_checkbox,
                self.interval_input,
                self.period_select,
                Empty(),
            ],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
            overflow="wrap",
        )

        apply_btn_container = Container([self.apply_btn], style="align-items: flex-end")

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked):
            if is_checked:
                self.interval_input.enable()
                self.period_select.enable()
            else:
                self.interval_input.disable()
                self.period_select.disable()

        return Container([self.description, settings_container, apply_btn_container])

    def get_automation_details(self) -> Tuple[int, str, int, str]:
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.interval_input.get_value()

        if not enabled:
            # removed = g.session.importer.unschedule_cloud_import()
            return None, None, None

        sec = get_seconds_from_period_and_interval(period, interval)
        if sec == 0:
            return None, None, None

        return sec, interval, period

    def save_automation_details(self, enabled: bool, interval: int, period: str) -> None:
        """
        :param enabled: Whether the automation is enabled.
        :param interval: Interval for synchronization.
        :param period: Period unit for synchronization (e.g., "minutes", "hours", "days").
        """
        if self.enabled_checkbox.is_checked() != enabled:
            if enabled:
                self.enabled_checkbox.check()
            else:
                self.enabled_checkbox.uncheck()
        if self.period_select.get_value() != period:
            self.period_select.set_value(period)
        if self.interval_input.get_value() != interval:
            self.interval_input.value = interval


# only SolutionCard/SolutionProject can be used as content
class SolutionCardNode(SolutionGraph.Node):
    def __new__(
        cls, content: Widget, x: int = 0, y: int = 0, *args, **kwargs
    ) -> SolutionGraph.Node:
        JinjaWidgets().incremental_widget_id_mode = True
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
