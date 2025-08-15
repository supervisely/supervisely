from typing import Literal, Optional, Union

from supervisely.app.widgets import Icons, SolutionCard
from supervisely.solution.base_node import SolutionCardNode, SolutionElement


class EmptyNode(SolutionElement):
    """
    Base class for nodes that can be used only as placeholders in the graph builder.
    It does not have any functionality and is used to structure the graph visually.
    """

    def __init__(
        self,
        title: str,
        description: str,
        x: int = 0,
        y: int = 0,
        width: int = 250,
        icon: Optional[Union[Icons, str]] = None,
        tooltip_position: Literal["left", "right"] = "right",
        badge: SolutionCard.Badge = None,
        *args,
        **kwargs,
    ):
        self.title = title
        self.description = description
        self.width = width
        if isinstance(icon, str):
            icon = Icons(class_name=icon, color="#4CAF50", bg_color="#E8F5E9")
        self.icon = icon
        self.tooltip_position = tooltip_position
        self.card = self._create_card()
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        super().__init__(*args, **kwargs)

        if badge is not None:
            if not isinstance(badge, SolutionCard.Badge):
                raise TypeError("Badge must be an instance of SolutionCard.Badge")
            self.card.add_badge(badge)

    def _create_card(self) -> SolutionCard:
        return SolutionCard(
            title=self.title,
            tooltip=self._create_tooltip(),
            width=self.width,
            tooltip_position=self.tooltip_position,
            icon=self.icon,
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        return SolutionCard.Tooltip(description=self.description)
