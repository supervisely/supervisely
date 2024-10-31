import json
from pathlib import Path
from typing import List, Optional, Union

from jinja2 import Template

from supervisely.api.annotation_api import AnnotationInfo
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import GridGalleryV2
from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget
from supervisely.project.project_meta import ProjectMeta


class GalleryWidget(BaseWidget):
    def __init__(
        self,
        name: str,
        filters: Optional[List] = None,
        is_modal: Optional[bool] = False,
        columns_number: int = 3,
    ):
        super().__init__(name)
        self.reference = self.id
        self.is_modal = is_modal
        self.click_handled = False
        self.click_gallery_id = None
        self.click_data = None
        self.click_gallery_items_limit = None
        self.image_left_header = False
        self._project_meta = None
        self.show_all_button = False
        self.columns_number = columns_number

        filters = filters  # or [{"confidence": [0.6, 1]}]
        self._gallery = GridGalleryV2(
            columns_number=columns_number,
            annotations_opacity=0.4,
            border_width=4,
            enable_zoom=False,
            default_tag_filters=filters,
            show_zoom_slider=False,
        )

    def set_project_meta(self, project_meta: ProjectMeta):
        self._project_meta = project_meta

    def set_images(
        self,
        image_infos: List[ImageInfo],
        ann_infos: List[AnnotationInfo],
        project_metas: List[ProjectMeta] = None,
        skip_tags_filtering: Optional[List[Union[bool, List[str]]]] = None,
    ):
        """One time operation"""
        if project_metas is None:
            project_metas = [self._project_meta] * self.columns_number

        if skip_tags_filtering is None:
            skip_tags_filtering = [False] * self.columns_number

        for idx, (image, ann) in enumerate(zip(image_infos, ann_infos)):
            image_name = image.name
            image_url = image.full_storage_url
            self._gallery.append(
                title=image_name,
                image_url=image_url,
                annotation_info=ann,
                column_index=idx % self.columns_number,
                project_meta=project_metas[idx % self.columns_number],
                ignore_tags_filtering=skip_tags_filtering[idx % self.columns_number],
            )

    def _get_init_data(self):
        res = {}
        self._gallery._update_filters()
        res.update(self._gallery.get_json_state())
        res.update(self._gallery.get_json_data()["content"])
        res["layoutData"] = res.pop("annotations")
        res["projectMeta"] = self._project_meta.to_json()
        return res

    def save_data(self, basepath: str) -> None:
        # init data
        basepath = basepath.rstrip("/")
        ensure_base_path(basepath + self.data_source)

        with open(basepath + self.data_source, "w") as f:
            json.dump(self._get_init_data(), f)

        # click data
        if self.click_data is not None:
            ensure_base_path(basepath + self.click_data_source)
            with open(basepath + self.click_data_source, "w") as f:
                json.dump(self.click_data, f)

    def get_state(self) -> None:
        return {}

    def to_html(self) -> str:
        template_str = Path(__file__).parent / "template.html"
        return Template(template_str.read_text()).render(self._get_template_data())

    def add_on_click(self, gallery_id, click_data, gallery_items_limit):
        self.click_handled = True
        self.click_gallery_id = gallery_id
        self.click_data = click_data
        self.click_gallery_items_limit = gallery_items_limit

    def add_image_left_header(self, html: str):
        self.image_left_header = html

    def _get_template_data(self):
        return {
            "widget_id": self.id,
            "reference": self.reference,  # TODO: same as id?
            "init_data_source": self.data_source,
            "is_modal": str(self.is_modal).lower(),
            "click_handled": self.click_handled,
            "show_all_button": self.show_all_button,
            "click_data_source": self.click_data_source,
            "click_gallery_id": self.click_gallery_id,
            "click_gallery_items_limit": self.click_gallery_items_limit,
            "image_left_header": self.image_left_header,
        }
