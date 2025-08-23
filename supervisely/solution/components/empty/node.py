import re
from typing import Literal, Optional, Union

from supervisely.app.widgets import Icons, SolutionCard
from supervisely.solution.base_node import BaseCardNode


class EmptyNode(BaseCardNode):
    """
    Base class for nodes that can be used only as placeholders in the graph builder.
    It does not have any functionality and is used to structure the graph visually.
    """

    def __init__(
        self,
        title: str = "Empty Node",
        description: str = None,
        width: int = 250,
        icon: str = "mdi mdi-google-downasaur",
        icon_color: str = "#1976D2",
        icon_bg_color: str = "#E3F2FD",
        tooltip_position: Literal["left", "right"] = "right",
        badge: SolutionCard.Badge = None,
        *args,
        **kwargs,
    ):
        if isinstance(icon, Icons):
            icon = re.sub(r'<i class="(.*?)"', r"\1", icon) if icon else None
        title = kwargs.pop("title", title)
        description = kwargs.pop("description", description)
        icon = kwargs.pop("icon", icon)
        icon_color = kwargs.pop("icon_color", icon_color)
        icon_bg_color = kwargs.pop("icon_bg_color", icon_bg_color)
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        if badge is not None:
            if not isinstance(badge, SolutionCard.Badge):
                raise TypeError("Badge must be an instance of SolutionCard.Badge")
            self.card.add_badge(badge)
