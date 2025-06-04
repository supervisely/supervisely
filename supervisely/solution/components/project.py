from typing import List, Optional, Union

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.app.widgets import (
    Button,
    SolutionCard,
    SolutionGraph,
    SolutionProject,
    Widget,
)
from supervisely.solution.components.card import Card


class ProjectNode(Card):
    card_cls = SolutionProject

    def __init__(
        self,
        x: int,
        y: int,
        project: ProjectInfo,
        title: Optional[str] = None,
        is_training: bool = False,
        tooltip_content: Optional[Widget] = None,
        show_qa_link: bool = True,
        show_project_link: bool = True,
        tooltip_position: str = "left",
        description: Optional[str] = None,
    ):
        self._api = Api.from_env()
        self.x = x
        self.y = y
        self.title = title
        self.project = project
        self.is_training = is_training
        stats_url = self.project.url.replace("datasets", "stats/datasets")
        tooltip_widgets = []
        if show_project_link:
            project_link = Button(
                "Open project",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=self.project.url,
                button_type="text",
            )
            tooltip_widgets.append(project_link)
        if show_qa_link:
            qa_link = Button(
                "QA stats",
                icon="zmdi zmdi-open-in-new",
                button_size="mini",
                plain=True,
                link=stats_url,
                button_type="text",
            )
            tooltip_widgets.append(qa_link)

        if tooltip_content is not None:
            if isinstance(tooltip_content, Widget):
                tooltip_content = [tooltip_content]
            for widget in tooltip_content:
                if isinstance(widget, Button):
                    widget.plain = True
                    widget.button_type = "text"
            tooltip_widgets.extend(tooltip_content)

        items_count = [project.items_count]
        preview_url = [project.image_preview_url]
        if project.items_count is None or project.items_count == 0:
            items_count = [0]
            preview_url = [None]

        if self.is_training:
            items_count, preview_url = [0, 0], [None, None]

        self.card = SolutionProject(
            title=self.title,
            preview_url=preview_url,
            items_count=items_count,
            project_id=self.project.id,
            width=270 if self.is_training else 180,
            tooltip=SolutionCard.Tooltip(
                description=description,
                content=tooltip_widgets,
            ),
            tooltip_position=tooltip_position,
        )
        self._node = SolutionGraph.Node(x=x, y=y, content=self.card)

    def _get_preview_details(self, ids: List[int]):
        preview_urls = []
        image_infos = self._api.image.get_info_by_id_batch(ids)
        for image in image_infos:
            preview_urls.append(image.preview_url)
        return preview_urls

    def update_preview(self, imgs: List[Union[int, str]], counts: List[int]):
        if all(isinstance(i, int) for i in imgs):
            imgs = self._get_preview_details(imgs)
        self.card.update_preview_url(imgs)
        self.card.update_items_count(counts)

    def update(
        self,
        project: ProjectInfo = None,
        new_items_count: int = None,
        urls: List[Union[int, str, None]] = None,
        counts: List[Union[int, None]] = None,
    ):
        if project is not None:
            self.project = project
        if new_items_count is not None:
            self.update_property(key="Last update", value=f"+{new_items_count}")
            self.update_property(key="Total", value=f"{self.project.items_count} images")
            self.update_badge_by_key(key="Last update:", label=f"+{new_items_count}")

        if self.is_training and urls is not None and counts is not None:
            self.update_preview(urls, counts)
        else:
            self.update_preview(
                [self.project.image_preview_url],
                [self.project.items_count],
            )
