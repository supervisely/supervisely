from typing import Literal, Optional, Union

from supervisely._utils import abs_url, is_development
from supervisely.app.widgets import Icons
from supervisely.solution.base_node import BaseCardNode


class LinkNode(BaseCardNode):
    """
    Base class for nodes that can be used as links to external resources.
    """

    def __init__(
        self,
        title: str,
        description: str,
        link: str = "",
        width: int = 250,
        icon: Optional[Union[Icons, str]] = None,
        icon_color: str = "#1976D2",
        icon_bg_color: str = "#E3F2FD",
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        if isinstance(icon, Icons):
            icon = icon.get_class_name()

        if link and link.startswith("/") and is_development():
            link = abs_url(link)
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            link=link,
            *args,
            **kwargs,
        )
