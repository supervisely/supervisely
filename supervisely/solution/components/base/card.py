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
        """Disables the card widget."""
        self.content.disable()
        super().disable()

    def enable(self):
        """Enables the card widget."""
        self.content.enable()
        super().enable()

    # ------------------------------------------------------------------
    # Tooltip Methods --------------------------------------------------
    # ------------------------------------------------------------------
    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        """
        Updates the property of the card.

        :param key: Key of the property.
        :type key: str
        :param value: Value of the property.
        :type value: str
        :param link: Link of the property.
        :type link: str
        :param highlight: Whether to highlight the property.
        :type highlight: bool
        """
        for prop in self.content.tooltip_properties:
            if prop["key"] == key:
                self.content.update_property(key, value, link, highlight)
                return
        self.content.add_property(key, value, link, highlight)

    def remove_property_by_key(self, key: str):
        """
        Removes the property by key of the card.

        :param key: Key of the property.
        :type key: str
        """
        self.content.remove_property_by_key(key)

    # ------------------------------------------------------------------
    # Badge Methods ----------------------------------------------------
    # ------------------------------------------------------------------
    def add_badge(self, badge):
        """
        Adds a badge to the card.

        :param badge: Badge to add.
        :type badge: Badge
        """
        self.content.add_badge(badge)

    def remove_badge(self, idx: int):
        """
        Removes the badge by index of the card.

        :param idx: Index of the badge.
        :type idx: int
        """
        self.content.remove_badge(idx)

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: Literal["info", "success", "warning", "error"] = "info",
    ):
        """
        Updates the badge by index of the card.

        :param idx: Index of the badge.
        :type idx: int
        :param label: Label of the badge.
        :type label: str
        :param on_hover: On hover text of the badge.
        :type on_hover: str
        :param badge_type: Type of the badge.
        :type badge_type: Literal["info", "success", "warning", "error"]
        """
        self.content.update_badge(idx, label, on_hover, badge_type)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        new_key: str = None,
        plain: Optional[bool] = None,
    ):
        """
        Updates the badge by key of the card.

        :param key: Key of the badge.
        :type key: str
        :param label: Label of the badge.
        :type label: str
        """
        self.content.update_badge_by_key(
            key=key,
            label=label,
            new_key=new_key,
            badge_type=badge_type,
            plain=plain,
        )

    def remove_badge_by_key(self, key: str):
        """
        Removes the badge by key from the card.

        :param key: Key of the badge.
        :type key: str
        """
        self.content.remove_badge_by_key(key)

    # ------------------------------------------------------------------
    # Automation Badge Methods -----------------------------------------
    # ------------------------------------------------------------------
    def update_automation_badge(self, enable: bool) -> None:
        """
        Updates the automation badge of the card.

        :param enable: Whether to enable the automation.
        :type enable: bool
        """
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
        """
        Updates the card to show that the main task is in progress.

        :param key: Key of the badge.
        :type key: Optional[str]
        """
        key = key or "⏳"
        self.content.update_badge_by_key(key=key, label="In progress", badge_type="info")

    def hide_in_progress_badge(self, key: Optional[str] = None):
        """
        Hides the in-progress badge from the card.

        :param key: Key of the badge.
        :type key: Optional[str]
        """
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

    # ------------------------------------------------------------------
    # Preview Methods --------------------------------------------------
    # ------------------------------------------------------------------
    def update_preview(self, imgs: List[str], counts: List[int]):
        """
        Updates the preview of the project.

        :param imgs: List of image urls.
        :type imgs: List[str]
        :param counts: List of counts.
        :type counts: List[int]
        """
        self.content.update_preview_url(imgs)
        self.content.update_items_count(counts)

    def update_preview_url(self, imgs: List[str]):
        """
        Updates the preview url of the project.

        :param imgs: List of image urls.
        :type imgs: List[str]
        """
        self.content.update_preview_url(imgs)

    def update_items_count(self, counts: List[int]):
        """
        Updates the items count of the project.

        :param counts: List of counts.
        :type counts: List[int]
        """
        self.content.update_items_count(counts)

    # ------------------------------------------------------------------
    # Update Methods ---------------------------------------------------
    # ------------------------------------------------------------------
    def update(
        self,
        project: ProjectInfo = None,
        new_items_count: int = None,
        urls: List[Union[int, str, None]] = None,
        counts: List[Union[int, None]] = None,
    ):
        """
        Updates the node with the new project.

        :param project: Project info.
        :type project: ProjectInfo
        :param new_items_count: New items count.
        :type new_items_count: int
        :param urls: List of image urls.
        :type urls: List[Union[int, str, None]]
        :param counts: List of counts.
        :type counts: List[Union[int, None]]
        """
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
