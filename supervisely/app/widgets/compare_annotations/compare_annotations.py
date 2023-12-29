from typing import List, Optional

from supervisely import Annotation
from supervisely.app.widgets import GridGallery


class CompareAnnotations(GridGallery):
    """CompareAnnotations is a simple widget that allows you to compare different annotations for one image.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/compare-data/compareannotations>`_
        (including screenshots and examples).

    :param columns_number: number of columns in the grid
    :type columns_number: int
    :param default_opacity: default opacity for annotations
    :type default_opacity: float
    :param fill_rectangle: if True, rectangles will be filled with color
    :type fill_rectangle: bool
    :param border_width: border width for rectangles
    :type border_width: int
    :param view_height: height of the widget
    :type view_height: int
    :param empty_message: message to show when there is no data
    :type empty_message: str
    :param widget_id: An unique identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import CompareAnnotations

        compare_annotations = CompareAnnotations(columns_number=2)

        # Setting the URL of the image to show
        compare_annotations.set_image_url("https://i.imgur.com/2Yj2xYh.jpg")

        # Adding annotations to the widget
        compare_annotations.append(ann1, "Annotation 1")
        compare_annotations.append(ann2, "Annotation 2")

    """

    def __init__(
        self,
        columns_number: int,
        default_opacity: Optional[float] = 0.5,
        fill_rectangle: Optional[bool] = True,
        border_width: Optional[int] = 3,
        view_height: Optional[int] = None,
        empty_message: Optional[str] = "No image was provided",
        widget_id: Optional[str] = None,
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
        """Returns url of the image to show.

        :return: url of the image to show
        :rtype: str
        """
        return self._image_url

    def set_image_url(self, image_url: str) -> None:
        """Sets url of the image to show.

        :param image_url: url of the image to show
        :type image_url: str
        """
        self._image_url = image_url

    def append(
        self,
        annotation: Optional[Annotation] = None,
        title: Optional[str] = "",
        column_index: Optional[int] = None,
    ) -> None:
        """Adds annotation to the widget.

        :param annotation: annotation to add
        :type annotation: Annotation, optional
        :param title: title for the annotation
        :type title: str, optional
        :param column_index: index of the column to add annotation to
        :type column_index: int, optional
        """
        super().append(self._image_url, annotation, title, column_index)

    def is_empty(self) -> bool:
        """Returns True if there is no image to show, False otherwise.

        :return: True if there is no image to show, False otherwise
        :rtype: bool
        """
        return self._image_url is None or self._image_url == ""

    def clean_up(self) -> None:
        """Removes all annotations and image url from the widget."""
        super().clean_up()
        self._image_url = None
        self._annotations = []
        self._titles = []

    def _set_items(self, annotations: List[Annotation]) -> None:
        for ann in annotations:
            self.append(self._image_url, ann)
