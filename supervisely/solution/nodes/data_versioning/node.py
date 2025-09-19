from typing import Optional

from supervisely.solution.components.link_node.node import LinkNode


class DataVersioningNode(LinkNode):
    """Node for linking to the Project Versions dashboard."""

    TITLE = "Data Versioning"
    DESCRIPTION = "Versioning allows you to track changes in your projects over time. Each version is a snapshot of the project at a specific point in time, enabling you to revert to previous versions if needed. Training sessions automatically create new versions, capturing the exact state of the project at that moment. You can track and manage these versions in the Versions tab of the project (or click this card to open it)."
    ICON = "mdi mdi-history"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(self, project_id: Optional[int] = None, *args, **kwargs):
        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        link = f"/projects/{project_id}/versions" if project_id is not None else ""
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
                "id": "data_versioning_project_id",
                "type": "target",
                "position": "top",
                "label": "Input",
                "connectable": True,
            },
            {
                "id": "data_versioning_output",
                "type": "source",
                "position": "bottom",
                "label": "Output",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self):
        return {
            "data_versioning_project_id": self.set_project_id,
        }

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------
    def set_project_id(self, project_id: Optional[int] = None):
        """Set project ID and update the link accordingly."""
        link = f"/projects/{project_id}/versions" if project_id is not None else ""
        self.set_link(link)
