from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Pagination(Widget):
    class Routes:
        CURRENT_CHANGE = "current_change"
        SIZE_CHANGE = "size_change"

    def __init__(
        self,
        total: int,
        page_size: int = 10,
        current_page: int = 1,
        layout: Union[
            str, List[Literal["sizes", "prev", "pager", "next", "jumper", "->", "total"]]
        ] = "prev, pager, next",
        compact: bool = False,
        page_size_options: List[int] = [10, 20, 30, 40, 50, 100],
        widget_id: str = None,
    ):
        self._total = total
        self._page_size = page_size
        self._current_page = current_page
        self._layout = ", ".join(layout) if type(layout) == list else layout
        self._compact = compact
        self._page_size_options = page_size_options

        self._current_change = False
        self._size_change = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "layout": self._layout,
            "total": self._total,
            "compact": self._compact,
            "pageSizeOptions": self._page_size_options,
        }

    def get_json_state(self):
        return {
            "currentPage": self._current_page,
            "pageSize": self._page_size,
        }

    def get_current_page(self):
        return StateJson()[self.widget_id]["currentPage"]

    def set_current_page(self, page: int):
        max_page = self.get_total() // self.get_page_size() + 1
        if page > max_page:
            page = max_page
        if page < 1:
            page = 1
        if type(page) != int:
            raise TypeError("Current page value must be int")
        self._current_page = page
        StateJson()[self.widget_id]["currentPage"] = self._current_page
        StateJson().send_changes()

    def get_total(self):
        return DataJson()[self.widget_id]["total"]

    def set_total(self, value: int):
        if type(value) != int:
            raise TypeError("Total value must be int")
        self._total = value
        DataJson()[self.widget_id]["total"] = self._total
        DataJson().send_changes()

    def get_page_size(self):
        return StateJson()[self.widget_id]["pageSize"]

    def set_page_size(self, value: int):
        if type(value) != int:
            raise TypeError("Page size value must be int")

        if value not in self.get_page_size_options():
            raise ValueError("Page size value must be in page size options")

        self._current_page = value
        StateJson()[self.widget_id]["pageSize"] = self._current_page
        StateJson().send_changes()

    def get_page_size_options(self):
        return DataJson()[self.widget_id]["pageSizeOptions"]

    def set_page_size_options(self, value: List[int]):
        if type(value) != list or not all(type(x) == int for x in value):
            raise TypeError("Page size options value must be list of integers")
        self._page_size_options = value
        DataJson()[self.widget_id]["pageSizeOptions"] = self._page_size_options
        DataJson().send_changes()

    def set_layout(
        self,
        value: Union[
            str, List[Literal["sizes", "prev", "pager", "next", "jumper", "->", "total", "slot"]]
        ],
    ):
        if type(value) == list:
            value = ", ".join(value)
        if type(value) != str:
            raise TypeError("Layout value must be str or list")
        self._layout = value
        DataJson()[self.widget_id]["layout"] = self._layout
        DataJson().send_changes()

    def page_changed(self, func):
        route_path = self.get_route_path(Pagination.Routes.CURRENT_CHANGE)
        server = self._sly_app.get_server()
        self._current_change = True

        @server.post(route_path)
        def _click():
            res = self.get_current_page()
            func(res)

        return _click

    def page_size_changed(self, func):
        route_path = self.get_route_path(Pagination.Routes.SIZE_CHANGE)
        server = self._sly_app.get_server()
        self._size_change = True

        @server.post(route_path)
        def _click():
            res = self.get_page_size()
            func(res)

        return _click
