import copy
from typing import Optional
import uuid

from supervisely.annotation.annotation import Annotation
from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets import GridGallery


class LabeledImage(GridGallery):
    """Single-image gallery view with annotation overlay, opacity slider, and zoom for detailed inspection."""

    def __init__(
        self,
        annotations_opacity: float = 0.5,
        show_opacity_slider: bool = True,
        enable_zoom: bool = False,
        resize_on_zoom: bool = False,
        fill_rectangle: bool = True,
        border_width: int = 3,
        view_height: Optional[int] = None,
        widget_id: str = None,
        empty_message: str = "No image was provided",
    ):
        """Initialize LabeledImage.

        :param annotations_opacity: Opacity of annotation overlays (0–1).
        :type annotations_opacity: float
        :param show_opacity_slider: If True, show opacity slider.
        :type show_opacity_slider: bool
        :param enable_zoom: If True, enable zoom.
        :type enable_zoom: bool
        :param resize_on_zoom: If True, resize on zoom.
        :type resize_on_zoom: bool
        :param fill_rectangle: If True, fill rectangles.
        :type fill_rectangle: bool
        :param border_width: Border width for shapes.
        :type border_width: int
        :param view_height: Fixed view height. None for auto.
        :type view_height: int, optional
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        :param empty_message: Message when no image.
        :type empty_message: str
        """
        self._image_id = None
        super().__init__(
            columns_number=1,
            annotations_opacity=annotations_opacity,
            show_opacity_slider=show_opacity_slider,
            enable_zoom=enable_zoom,
            resize_on_zoom=resize_on_zoom,
            fill_rectangle=fill_rectangle,
            border_width=border_width,
            view_height=view_height,
            empty_message=empty_message,
            widget_id=widget_id,
        )

    def set(
        self,
        title,
        image_url,
        ann: Annotation = None,
        image_id=None,
        zoom_to=None,
        zoom_factor=1.2,
        title_url=None,
        force_clean_up=False,
    ):
        self.clean_up()
        if force_clean_up:
            # if image url is the same, we need to force clean up
            DataJson().send_changes()
        self.append(
            image_url=image_url,
            annotation=ann,
            title=title,
            zoom_to=zoom_to,
            zoom_factor=zoom_factor,
            title_url=title_url,
        )
        self._image_id = image_id
        DataJson().send_changes()

    def clean_up(self):
        super().clean_up()
        self._image_id = None

    def is_empty(self):
        return len(self._data) == 0

    @property
    def id(self):
        return self._image_id
