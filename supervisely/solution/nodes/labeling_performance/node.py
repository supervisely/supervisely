from typing import Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.solution.components.link_node.node import LinkNode


class LabelingQueuePerformanceNode(LinkNode):
    """
    Node for displaying a link to the Labeling Performance page of the Labeling Queue.
    """

    title = "Labeling Performance"
    description = "View the performance of the labeling queue."
    icon = "mdi mdi-chart-bar"
    icon_color = "#1976D2"
    icon_bg_color = "#E3F2FD"

    def __init__(
        self,
        queue_id: int = None,
        width: int = 250,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        api = Api.from_env()
        link = f"/labeling-performance/"
        if queue_id is not None:
            queue = api.labeling_queue.get_info_by_id(queue_id)
            if queue is not None:
                link += f"?jobs={queue.jobs}"

        title = kwargs.pop("title", self.title)
        description = kwargs.pop("description", self.description)
        icon = kwargs.pop("icon", self.icon)
        icon_color = kwargs.pop("icon_color", self.icon_color)
        icon_bg_color = kwargs.pop("icon_bg_color", self.icon_bg_color)
        super().__init__(
            title=title,
            description=description,
            link=link,
            width=width,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            tooltip_position=tooltip_position,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "labeling_performance",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self):
        return {
            "labeling_performance": self.set_queue_id,
        }

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_queue_id(self, queue_id: Optional[int] = None):
        """Set project ID and update the link accordingly."""
        api = Api.from_env()
        link = f"/labeling-performance/"
        if queue_id is not None:
            queue = api.labeling_queue.get_info_by_id(queue_id)
            if queue is not None:
                link += f"?jobs={queue.jobs}"
        # TODO: update link in the node
