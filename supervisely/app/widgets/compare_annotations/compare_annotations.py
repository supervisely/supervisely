from typing import Optional, List
from supervisely import Annotation
from supervisely.app.widgets import GridGallery


class CompareAnnotations(GridGallery):
    def __init__(
        self,
        columns_number: int,
        default_opacity: float = 0.5,
        fill_rectangle: bool = True,
        border_width: int = 3,
        view_height: Optional[int] = None,
        empty_message: str = "No image was provided",
        widget_id: str = None,
    ):
        self._image_url: str = None
        self._annotations: List[Annotation] = []
        self._titles: List[str] = []
        self._set_items(self._annotations)

        super().__init__(
            columns_number=columns_number,
            annotations_opacity=default_opacity,
            show_opacity_slider=True,
            enable_zoom=True,
            sync_views=True,
            fill_rectangle=fill_rectangle,
            border_width=border_width,
            show_preview=False,
            view_height=view_height,
            empty_message=empty_message,
            widget_id=widget_id,
        )

    @property
    def image_url(self) -> str:
        return self._image_url

    def set_image_url(self, image_url: str) -> None:
        self._image_url = image_url

    def append(
        self,
        annotation: Annotation = None,
        title: str = "",
        column_index: int = None,
    ):
        super().append(self._image_url, annotation, title, column_index)

    def is_empty(self) -> bool:
        return self._image_url == None or self._image_url == ""

    def clean_up(self) -> None:
        super().clean_up()
        self._image_url = None
        self._annotations = []
        self._titles = []

    def _set_items(self, annotations: List[Annotation]) -> None:
        for ann in annotations:
            self.append(self._image_url, ann)
