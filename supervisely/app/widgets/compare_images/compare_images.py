from supervisely.annotation.annotation import Annotation
from supervisely.app import DataJson
from supervisely.app.widgets import GridGallery


class CompareImages(GridGallery):
    def __init__(
        self,
        annotations_opacity: float = 0.5,
        show_opacity_slider: bool = True,
        enable_zoom: bool = False,
        resize_on_zoom: bool = False,
        fill_rectangle: bool = True,
        border_width: int = 3,
        widget_id: str = None,
    ):
        super().__init__(
            columns_number=2,
            annotations_opacity=annotations_opacity,
            show_opacity_slider=show_opacity_slider,
            enable_zoom=enable_zoom,
            resize_on_zoom=resize_on_zoom,
            fill_rectangle=fill_rectangle,
            border_width=border_width,
            widget_id=widget_id,
        )

    def _add_image(
        self,
        title,
        image_url,
        position_index: int,
        ann: Annotation = None,
        zoom_to: int = None,
        zoom_factor: float = 1.2,
        title_url=None,
    ):
        exist_data = None
        for curr_data in self._data:
            if curr_data["column_index"] == position_index:
                exist_data = curr_data

        self.clean_up()
        super().append(
            image_url=image_url,
            annotation=ann,
            title=title,
            column_index=position_index ^ 1,
            zoom_to=zoom_to,
            zoom_factor=zoom_factor,
            title_url=title_url,
        )
        if exist_data is not None:
            super().append(
                image_url=exist_data["image_url"],
                annotation=exist_data["annotation"],
                title=exist_data["title"],
                column_index=position_index,
                zoom_to=exist_data["zoom_to"],
                zoom_factor=exist_data["zoom_factor"],
                title_url=exist_data["title_url"],
            )
        DataJson().send_changes()

    def set_left(
        self,
        title,
        image_url,
        ann: Annotation = None,
        zoom_to: int = None,
        zoom_factor: float = 1.2,
        title_url=None,
    ):
        self._add_image(
            title=title,
            position_index=1,
            image_url=image_url,
            ann=ann,
            zoom_to=zoom_to,
            zoom_factor=zoom_factor,
            title_url=title_url,
        )

    def set_right(
        self,
        title,
        image_url,
        ann: Annotation = None,
        zoom_to: int = None,
        zoom_factor: float = 1.2,
        title_url=None,
    ):
        self._add_image(
            title=title,
            position_index=0,
            image_url=image_url,
            ann=ann,
            zoom_to=zoom_to,
            zoom_factor=zoom_factor,
            title_url=title_url,
        )

    def append(
        self,
        image_url: str,
        annotation: Annotation = None,
        title: str = "",
        column_index: int = None,
        zoom_to: int = None,
        zoom_factor: float = 1.2,
        title_url=None,
    ):
        raise ValueError("Use only 'set_left' or 'set_right' methods to add data")
