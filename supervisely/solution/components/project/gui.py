from typing import List, Optional

from supervisely._utils import abs_url
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import Button, SolutionProject


class ProjectGUI:
    def __init__(
        self,
        title: str,
        project: ProjectInfo,
        is_training: bool = False,
        dataset: Optional[DatasetInfo] = None,
    ):
        self.title = title
        self.project = project
        self.is_training = is_training
        self.dataset = dataset
        self.card = self._create_card()

    # ------------------------------------------------------------------
    # GUI --------------------------------------------------------------
    # ------------------------------------------------------------------
    def _create_card(self) -> SolutionProject:
        """Creates the SolutionProject card with appropriate settings"""
        if self.is_training:
            items_count, preview_url = [0, 0], [None, None]
        else:
            items_count = (
                self.project.items_count if self.dataset is None else self.dataset.items_count
            )
            preview_url = (
                self.project.image_preview_url
                if self.dataset is None
                else self.dataset.image_preview_url
            )
            items_count, preview_url = [items_count or 0], [preview_url]
            if preview_url[0] == abs_url("/"):
                preview_url[0] = None

        return SolutionProject(
            title=self.title,
            preview_url=preview_url,
            items_count=items_count,
            project_id=self.project.id,
            width=270 if self.is_training else 180,
            tooltip=self._create_tooltip(),
            tooltip_position="left",
        )

    def _create_tooltip(self) -> SolutionProject.Tooltip:
        """Creates the tooltip for the SolutionProject card"""
        stats_url = self.project.url.replace("datasets", "stats/datasets")
        tooltip_widgets = [
            Button(
                "Open project",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=self.project.url,
                button_type="text",
            ),
            Button(
                "QA stats",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=stats_url,
                button_type="text",
            ),
        ]

        return SolutionProject.Tooltip(
            description="Each import creates a dataset folder in the Input Project, centralising all incoming data and easily managing it over time.",
            content=tooltip_widgets,
        )

    # ------------------------------------------------------------------
    # Update Methods ---------------------------------------------------
    # ------------------------------------------------------------------
    def update_preview(self, preview_urls: List[str], items_counts: List[int]) -> None:
        """Update preview URLs and item counts for the project card"""
        self.card.update_preview_url(preview_urls)
        self.card.update_items_count(items_counts)
