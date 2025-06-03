from typing import Dict, List, Literal, Optional

from supervisely.app.widgets import Button, SolutionsCard, SolutionsGraph, Widget


class Card:
    card_cls = SolutionsCard

    def __init__(
        self,
        x: int,
        y: int,
        title: Optional[str] = None,
        content: Optional[List[Widget]] = None,
        tooltip_widgets: Optional[List[Widget]] = None,
        description: Optional[str] = None,
        properties: Optional[List[Dict]] = None,
        tooltip_position: Literal["left", "right"] = "left",
        link: Optional[str] = None,
        width: int = 250,
    ):
        self.x = x
        self.y = y
        self.link = link
        tooltip = None
        if tooltip_widgets is not None:
            if isinstance(tooltip_widgets, Widget):
                tooltip_widgets = [tooltip_widgets]
            for widget in tooltip_widgets:
                if isinstance(widget, Button):
                    widget.plain = True
                    widget.button_type = "text"
        if tooltip_widgets is not None or description is not None or properties is not None:
            tooltip = SolutionsCard.Tooltip(
                description=description,
                content=tooltip_widgets,
                properties=properties,
            )
        self.card = SolutionsCard(
            title=title,
            content=content,
            tooltip=tooltip,
            tooltip_position=tooltip_position,
            link=link,
            width=width,
        )

        self._node = SolutionsGraph.Node(x=self.x, y=self.y, content=self.card)

    @property
    def node(self):
        return self._node

    def update_property(self, key: str, value: str, link: str = None, highlight: bool = None):
        for prop in self.card.tooltip_properties:
            if prop["key"] == key:
                self.card.update_property(key, value, link, highlight)
                return
        # if not found, add new property
        self.card.add_property(key, value, link, highlight)

    def remove_property_by_key(self, key: str):
        self.card.remove_property_by_key(key)

    def update_badge(
        self,
        idx: int,
        label: str,
        on_hover: str = None,
        badge_type: Literal["info", "success", "warning", "error"] = "info",
    ):
        self.card.update_badge(idx, label, on_hover, badge_type)

    def update_badge_by_key(
        self,
        key: str,
        label: str,
        badge_type: Literal["info", "success", "warning", "error"] = None,
        new_key: str = None,
    ):
        for idx, prop in enumerate(self.card.badges):
            if prop["on_hover"] == key:
                self.card.update_badge(idx, label, new_key, badge_type)
                return
        # if not found, add new badge
        self.card.add_badge(
            self.card_cls.Badge(
                label=label,
                on_hover=new_key or key,
                badge_type=badge_type or "info",
            )
        )

    def add_badge(self, badge: card_cls.Badge):
        self.card.add_badge(badge)

    def remove_badge(self, idx: int):
        self.card.remove_badge(idx)

    def remove_badge_by_key(self, key: str):
        self.card.remove_badge_by_key(key)

    def update_automation_badge(self, enable: bool) -> None:
        for idx, prop in enumerate(self.card.badges):
            if prop["on_hover"] == "Automation":
                if enable:
                    pass  # already enabled
                else:
                    self.card.remove_badge(idx)
                return

        if enable:  # if not found
            self.card.add_badge(
                self.card_cls.Badge(
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
