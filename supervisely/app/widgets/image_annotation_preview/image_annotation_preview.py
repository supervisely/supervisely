from supervisely.app.widgets_context import JinjaWidgets

from supervisely.annotation.annotation import Annotation, ProjectMeta
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class ImageAnnotationPreview(Widget):
    def __init__(
        self,
        annotations_opacity: float = 0.5,
        enable_zoom: bool = False,
        line_width: int = 1,
        widget_id: str = None,
    ):
        self._image_url = None
        self._annotation = None
        self._project_meta = None
        self._opacity = annotations_opacity
        self._enable_zoom = enable_zoom
        self._line_width = line_width

        super().__init__(widget_id=widget_id, file_path=__file__)
        script_path = "./sly/css/app/widgets/image_annotation_preview/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

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
                "lineWidth": self._line_width,
                "fitOnResize": True,
            }
        }

    def set(
        self,
        image_url,
        ann: Annotation = None,
        project_meta: ProjectMeta = None,
    ):
        self.clean_up()
        self._image_url = image_url
        self._annotation = ann
        self._project_meta = project_meta
        self.update_data()
        DataJson().send_changes()

    def clean_up(self):
        self._image_shape = None
        self._image_url = None
        self._annotation = None
        self._project_meta = None
        self.update_data()
        DataJson().send_changes()

    def is_empty(self):
        return self._image_url is None
