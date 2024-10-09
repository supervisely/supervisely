import json
from typing import List, Optional

from supervisely.api.annotation_api import AnnotationInfo
from supervisely.api.image_api import ImageInfo
from supervisely.app.widgets import GridGalleryV2
from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget
from supervisely.project.project_meta import ProjectMeta


class GalleryWidget(BaseWidget):
    def __init__(self, name: str, filters: Optional[List] = None, is_modal: Optional[str] = False):
        super().__init__(name)
        self.reference = self.id
        self.init_data_source = f"/data/{self.name}_{self.id}.json"
        self.is_modal = is_modal
        self.click_hadled = False
        self.click_data_source = f"/data/{self.name}_{self.id}_click_data.json"
        self.click_gallery_id = None
        self.click_data = None
        self.click_gallery_items_limit = None
        self.image_left_header = None

        filters = filters or [{"confidence": [0.6, 1]}]
        self._gallery = GridGalleryV2(
            columns_number=3,
            annotations_opacity=0.4,
            border_width=4,
            enable_zoom=False,
            default_tag_filters=filters,
            show_zoom_slider=False,
        )

    def set_images(
        self,
        image_infos: List[ImageInfo],
        ann_infos: List[AnnotationInfo],
        project_meta: ProjectMeta,
    ):
        """One time operation"""
        self._gallery._project_meta = project_meta
        for idx, (pred_image, ann_info) in enumerate(zip(image_infos, ann_infos)):
            image_name = pred_image.name
            image_url = pred_image.full_storage_url
            self._gallery.append(
                title=image_name,
                image_url=image_url,
                annotation_info=ann_info,
                column_index=idx % 3,
                project_meta=project_meta,
                ignore_tags_filtering=["outcome"],
            )

    def _get_init_data(self):
        res = {}
        res.update(self._gallery.get_json_state())
        res.update(self._gallery.get_json_data()["content"])
        res["layoutData"] = res.pop("annotations")
        res["projectMeta"] = self._gallery._project_meta.to_json()
        return res

    def save_data(self, basepath: str) -> None:
        # init data
        basepath = basepath.rstrip("/")
        ensure_base_path(basepath + self.init_data_source)

        with open(basepath + self.init_data_source, "w") as f:
            json.dump(self._get_init_data(), f)

        # click data
        if self.click_data is not None:
            ensure_base_path(basepath + self.click_data_source)
            with open(basepath + self.click_data_source, "w") as f:
                json.dump(self.click_data, f)

    def add_on_click(self, gallery_id, click_data, gallery_items_limit):
        self.click_hadled = True
        self.click_gallery_id = gallery_id
        self.click_data = click_data
        self.click_gallery_items_limit = gallery_items_limit

    def add_image_left_header(self, html: str):
        self.image_left_header = html

    def _get_template_data(self):
        return {
            "widget_id": self.id,
            "reference": self.reference,  # TODO: same as id?
            "init_data_source": self.init_data_source,
            "is_modal": self.is_modal,
            "click_hadnled": self.click_hadled,
            "click_data_source": self.click_data_source,
            "click_gallery_id": self.click_gallery_id,
            "click_gallery_items_limit": self.click_gallery_items_limit,
            "image_left_header": self.image_left_header,
        }
