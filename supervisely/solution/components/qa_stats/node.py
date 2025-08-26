from typing import Optional

from supervisely.solution.components.link_node.node import LinkNode


class QAStatsNode(LinkNode):
    """Node for linking to the QA Stats dashboard."""

    title = "QA Stats"
    description = "View detailed quality assurance statistics and insights for your project."
    icon = "mdi mdi-chart-bar"
    icon_color = "#1976D2"
    icon_bg_color = "#E3F2FD"

    def __init__(self, project_id: Optional[int] = None, *args, **kwargs):
        title = kwargs.pop("title", self.title)
        description = kwargs.pop("description", self.description)
        icon = kwargs.pop("icon", self.icon)
        icon_color = kwargs.pop("icon_color", self.icon_color)
        icon_bg_color = kwargs.pop("icon_bg_color", self.icon_bg_color)
        link = f"/projects/{project_id}/stats/datasets" if project_id is not None else ""
        link = kwargs.pop("link", link)

        self.project_id = project_id
        self.link = link
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
        # TODO: update link in the node
