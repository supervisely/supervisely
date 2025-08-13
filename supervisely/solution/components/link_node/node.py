from typing import Literal, Optional, Union

from supervisely._utils import abs_url, is_development
from supervisely.app.widgets import Icons
from supervisely.solution.base_node import SolutionCardNode, SolutionElement


class LinkNode(SolutionElement):
    """
    Base class for nodes that can be used as links to external resources.
    """

    def __init__(
        self,
        title: str,
        description: str,
        link: str = "",
        width: int = 250,
        x: int = 0,
        y: int = 0,
        icon: Optional[Union[Icons, str]] = None,
        icon_color: str = "#1976D2",
        icon_bg_color: str = "#E3F2FD",
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(icon, Icons):
            icon = icon.get_class_name()

        if link and link.startswith("/") and is_development():
            link = abs_url(link)

        self.card = self._build_card(
            title=title,
            tooltip_description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            link=link,
            width=width,
            tooltip_position=tooltip_position,
        )
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
