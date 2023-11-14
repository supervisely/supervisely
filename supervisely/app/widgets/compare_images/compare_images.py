import uuid
import time
from typing import List, Optional
from supervisely import Annotation
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget, LabeledImage, Image, ImageAnnotationPreview, GridGallery
from typing import Optional


class CompareImages(GridGallery):
    def __init__(
        self,
        columns_number: int,
        image_url: str = None,
        annotations: List[Annotation] = [],
        annotation_names: List[str] = [],
        annotations_opacity: int = 0.5,
        show_opacity_slider: bool = True,
        enable_zoom: bool = False,
        resize_on_zoom: bool = False,
        sync_views: bool = False,
        fill_rectangle: bool = True,
        border_width: int = 3,
        show_preview: bool = False,
        empty_message: str = "No image to display",
        widget_id: str = None,
    ):
        # self._validate_annotations(annotations)
        # self._validate_annotation_names(annotation_names)
        self._set_items(annotations)

        self._image_url = image_url
        self._annotations = annotations
        self._annotation_names = annotation_names
        self._columns_number = columns_number
        self._annotations_opacity = annotations_opacity
        self._show_opacity_slider = show_opacity_slider
        self._enable_zoom = enable_zoom
        self._resize_on_zoom = resize_on_zoom
        self._sync_views = sync_views
        self._fill_rectangle = fill_rectangle
        self._border_width = border_width
        self._show_preview = show_preview
        self._empty_message = empty_message
        self._widget_id = widget_id

        self._layout = []

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
            empty_message=empty_message,
            widget_id=widget_id,
        )

    def set_image_url(self, image_url: str):
        self._image_url = image_url

    def _set_items(self, annotations):
        for ann in annotations:
            self.append(self._image_url, ann)

    def append(
        self,
        annotation: Annotation = None,
        title: str = "",
        column_index: int = None,
    ):
        super().append(self._image_url, annotation, title, column_index)

    # def get_json_data(self):
    #     return {}

    # def get_json_state(self):
    #     return {}
