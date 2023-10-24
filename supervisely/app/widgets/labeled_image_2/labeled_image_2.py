import copy
from typing import Optional
import uuid

from supervisely.annotation.annotation import Annotation, ProjectMeta
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class LabeledImage2(Widget):
    def __init__(
        self,
        annotations_opacity: float = 0.5,
        enable_zoom: bool = False,
        border_width: int = 3,
        widget_id: str = None,
    ):
        self._image_url = None
        self._annotation = None
        self._project_meta = None
        self._opacity = annotations_opacity
        self._enable_zoom = enable_zoom
        self._border_width = border_width

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "imageUrl": self._image_url,
            "projectMeta": self._project_meta.to_json() if self._project_meta else None,
            "annotation": {"annotation": self._annotation.to_json()} if self._annotation else None,
        }

    def get_json_state(self):
        return {
            "options": {
                "enableZoom": self._enable_zoom,
                "opacity": self._opacity,
                "borderWidth": self._border_width,
            }
        }

    def set(
        self,
        image_url,
        ann: Annotation = None,
        project_meta: ProjectMeta = None,
    ):
        self._image_url = image_url
        self._annotation = ann
        self._project_meta = project_meta
        self.update_data()
        DataJson().send_changes()

    def clean_up(self):
        self.set(image_url=None, ann=None, project_meta=None)

    def is_empty(self):
        return self._image_url is None
