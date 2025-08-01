from typing import List, Literal, Optional, Union

from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import (
    SolutionCard,
    SolutionGraph,
    SolutionProject,
    Widget,
)
from supervisely.app.widgets_context import JinjaWidgets


# only SolutionCard/SolutionProject can be used as content
class SolutionCardNode(SolutionGraph.Node):
    def __new__(
        cls, content: Widget, x: int = 0, y: int = 0, *args, **kwargs
    ) -> SolutionGraph.Node:
        JinjaWidgets().incremental_widget_id_mode = True
        if not isinstance(content, (SolutionCard, SolutionProject)):
            raise TypeError("Content must be one of SolutionCard or SolutionProject")
        return super().__new__(cls, *args, **kwargs)

    # ------------------------------------------------------------------
    # Base Widget Settings ---------------------------------------------
    # ------------------------------------------------------------------
    def disable(self):
        self.content.disable()
        super().disable()

    def enable(self):
        self.content.enable()
        super().enable()

    # ------------------------------------------------------------------
    # Tooltip Methods --------------------------------------------------
    # ------------------------------------------------------------------
    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        for prop in self.content.tooltip_properties:
            if prop["key"] == key:
                self.content.update_property(key, value, link, highlight)
                return
        self.content.add_property(key, value, link, highlight)

    def remove_property_by_key(self, key: str):
        self.content.remove_property_by_key(key)

    # ------------------------------------------------------------------
    # Badge Methods ----------------------------------------------------
    # ------------------------------------------------------------------
    def add_badge(self, badge):
        self.content.add_badge(badge)

    def remove_badge(self, idx: int):
        self.content.remove_badge(idx)

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

    def remove_badge_by_key(self, key: str):
        self.content.remove_badge_by_key(key)

    # ------------------------------------------------------------------
    # Automation Badge Methods -----------------------------------------
    # ------------------------------------------------------------------
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
        """Updates the card to show that automation is enabled."""
        self.update_automation_badge(True)

    def hide_automation_badge(self) -> None:
        """Updates the card to show that automation is disabled."""
        self.update_automation_badge(False)

    # ------------------------------------------------------------------
    # In Progress Badge Methods ----------------------------------------
    # ------------------------------------------------------------------
    def show_in_progress_badge(self, key: Optional[str] = None):
        """Updates the card to show that the main task is in progress."""
        key = key or "⏳"
        self.content.update_badge_by_key(key=key, label="In progress", badge_type="info")

    def hide_in_progress_badge(self, key: Optional[str] = None):
        """Hides the in-progress badge from the card."""
        key = key or "⏳"
        self.content.remove_badge_by_key(key=key)

    # ------------------------------------------------------------------
    # Finished Badge Methods -------------------------------------------
    # ------------------------------------------------------------------
    def show_finished_badge(self):
        """Updates the card to show that main task is finished."""
        self.content.update_badge_by_key(
            key="Finished", label="✅", plain=True, badge_type="success"
        )

    def hide_finished_badge(self):
        """Hides the finished badge from the card."""
        self.content.remove_badge_by_key(key="Finished")

    # ------------------------------------------------------------------
    # Failed Badge Methods ---------------------------------------------
    # ------------------------------------------------------------------
    def show_failed_badge(self):
        """Updates the card to show that the main task has failed."""
        self.content.update_badge_by_key(key="Failed", label="❌", plain=True, badge_type="error")

    def hide_failed_badge(self):
        """Hides the failed badge from the card."""
        self.content.remove_badge_by_key(key="Failed")


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
