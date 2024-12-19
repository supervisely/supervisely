from supervisely.app import DataJson
from supervisely.app.widgets import Widget
from typing import Optional, Union


class IFrame(Widget):
    def __init__(
        self,
        path_to_html: str = "",
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
        widget_id: str = None,
    ):
        self._path_to_html = path_to_html
        self._height, self._width = self._check_plot_size(height=height, width=width)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _check_plot_size(
        self,
        height: Optional[Union[int, str]],
        width: Optional[Union[int, str]],
    ):
        if height is None and width is None:
            return "auto", "100%"

        def _check_single_size(size: Optional[Union[int, str]]):
            if size is None:
                return "auto"
            elif isinstance(size, int):
                return f"{size}px"
            elif isinstance(size, str):
                if size.endswith("px") or size.endswith("%") or size == "auto":
                    return size
                else:
                    raise ValueError(f"size must be in pixels or percent, got '{size}'")
            else:
                raise ValueError(f"size must be int or str, got '{type(size)}'")

        height = _check_single_size(size=height)
        width = _check_single_size(size=width)
        return height, width

    def get_json_data(self):
        return {
            "pathToHtml": self._path_to_html,
            "height": self._height,
            "width": self._width,
        }

    def get_json_state(self):
        return None

    def set(
        self,
        path_to_html: str,
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
    ):
        height = height or self._height
        width = width or self._width
        self._update(path_to_html=path_to_html, height=height, width=width)

    def clean_up(self):
        self._update(path_to_html="", height=None, width=None)

    def set_plot_size(
        self,
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
    ):
        self._update(path_to_html=self._path_to_html, height=height, width=width)

    def _update(
        self,
        path_to_html: str = "",
        height: Optional[Union[int, str]] = None,
        width: Optional[Union[int, str]] = None,
    ):
        self._path_to_html = path_to_html
        self._height, self._width = self._check_plot_size(height=height, width=width)
        DataJson()[self.widget_id]["pathToHtml"] = self._path_to_html
        DataJson()[self.widget_id]["height"] = self._height
        DataJson()[self.widget_id]["width"] = self._width
        DataJson().send_changes()
