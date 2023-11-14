from typing import Optional
from supervisely import Annotation
from supervisely.app.widgets import GridGallery


class CompareAnnotations(GridGallery):
    def __init__(
        self,
        columns_number: int,
        annotations_opacity: float = 0.5,
        show_opacity_slider: bool = True,
        enable_zoom: bool = False,
        resize_on_zoom: bool = False,
        sync_views: bool = False,
        fill_rectangle: bool = True,
        border_width: int = 3,
        show_preview: bool = False,
        view_height: Optional[int] = None,
        empty_message: str = "No image was provided",
        widget_id: str = None,
    ):
        self._image_url = None
        self._annotations = []
        self._titles = []
        self._set_items(self._annotations)

        super().__init__(
            columns_number=columns_number,
            annotations_opacity=annotations_opacity,
            show_opacity_slider=show_opacity_slider,
            enable_zoom=enable_zoom,
            resize_on_zoom=resize_on_zoom,
            sync_views=sync_views,
            fill_rectangle=fill_rectangle,
            border_width=border_width,
            show_preview=show_preview,
            view_height=view_height,
            empty_message=empty_message,
            widget_id=widget_id,
        )

    @property
    def image_url(self):
        return self._image_url

    def set_image_url(self, image_url: str):
        self._image_url = image_url

    def append(
        self,
        annotation: Annotation = None,
        title: str = "",
        column_index: int = None,
    ):
        super().append(self._image_url, annotation, title, column_index)

    def is_empty(self):
        return self._image_url == None or self._image_url == ""

    def clean_up(self):
        super().clean_up()
        self._image_url = None
        self._annotations = []
        self._titles = []

    def _set_items(self, annotations):
        for ann in annotations:
            self.append(self._image_url, ann)
