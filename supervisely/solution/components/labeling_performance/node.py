from typing import Literal, Optional, Union

from supervisely._utils import abs_url, is_development
from supervisely.api.api import Api
from supervisely.app.widgets import Icons
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.components.link_node.node import LinkNode


class LabelingQueuePerformanceNode(LinkNode):
    """
    Node for displaying a link to the Labeling Performance page of the Labeling Queue.
    """

    def __init__(
        self,
        queue_id: int,
        title: str = "Labeling Performance",
        description: str = "View the performance of the labeling queue.",
        width: int = 250,
        x: int = 0,
        y: int = 0,
        icon: Optional[Union[Icons, str]] = "zmdi zmdi-chart",
        icon_color: str = "#1976D2",
        icon_bg_color: str = "#E3F2FD",
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        api = Api.from_env()
        link = f"/labeling-performance/"
        queue = api.labeling_queue.get_info_by_id(queue_id)
        if queue is not None:
            link += f"?jobs={queue.jobs}"
        if is_development():
            link = abs_url(link)
        super().__init__(
            title=title,
            description=description,
            link=link,
            width=width,
            x=x,
            y=y,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            tooltip_position=tooltip_position,
            *args,
            **kwargs,
        )
