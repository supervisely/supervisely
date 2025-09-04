from typing import Optional

from supervisely.solution.components.link_node.node import LinkNode


class QAStatsNode(LinkNode):
    """Node for linking to the QA Stats dashboard."""

    TITLE = "QA Stats"
    DESCRIPTION = "Open the QA & Stats page to explore detailed insights into the project."
    ICON = "mdi mdi-chart-bar"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(self, project_id: Optional[int] = None, *args, **kwargs):
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        link = f"/projects/{project_id}/stats/datasets" if project_id is not None else ""
        link = kwargs.pop("link", link)

        self.project_id = project_id
        self._click_handled = True
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

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "qa_stats_project_id",
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
            "qa_stats_project_id": self.set_project_id,
        }

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_project_id(self, project_id: Optional[int] = None):
        """Set project ID and update the link accordingly."""
        link = f"/projects/{project_id}/stats/datasets" if project_id is not None else ""
        self.set_link(link)
