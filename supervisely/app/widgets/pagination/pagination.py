from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget
from typing import List


class Pagination(Widget):
    class Routes:
        CURRENT_CHANGE = "current_change"
        SIZE_CHANGE = "size_change"

    def __init__(
        self,
        total: int,
        layout: str = "prev, pager, next, jumper, ->, total",
        current_page: int = 1,
        small: bool = False,
        page_size: int = 10,
        page_sizes: List[int] = [10, 20, 30, 40, 50, 100],
        widget_id: str = None,
    ):
        self._layout = layout
        self._total = total
        self._current_page = current_page
        self._small = small
        self._page_size = page_size
        self._page_sizes = page_sizes

        self._current_change = False
        self._size_change = False
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "layout": self._layout,
            "total": self._total,
            "small": self._small,
            "page_sizes": self._page_sizes,
        }

    def get_json_state(self):
        return {
            "current_page": self._current_page,
            "page_size": self._page_size,
        }

    def get_current_page(self):
        return StateJson()[self.widget_id]["current_page"]

    def get_page_size(self):
        return StateJson()[self.widget_id]["page_size"]

    def set_page_size(self, value: int):
        self._current_page = value
        StateJson()[self.widget_id]["current_page"] = self._current_page
        StateJson().send_changes()

    def set_current_size(self, value: int):
        self._page_size = value
        StateJson()[self.widget_id]["page_size"] = self._page_size
        StateJson().send_changes()

    def get_layout(self):
        return DataJson()[self.widget_id]["layout"]

    def set_layout(self, value: str):
        self._layout = value
        DataJson()[self.widget_id]["layout"] = self._layout
        DataJson().send_changes()

    def get_total(self):
        return DataJson()[self.widget_id]["total"]

    def set_total(self, value: int):
        self._total = value
        DataJson()[self.widget_id]["total"] = self._total
        DataJson().send_changes()

    def unable_small_pagination(self):
        self._small = True
        DataJson()[self.widget_id]["small"] = self._small
        DataJson().send_changes()

    def disable_small_pagination(self):
        self._small = False
        DataJson()[self.widget_id]["small"] = self._small
        DataJson().send_changes()

    def get_page_sizes(self):
        return DataJson()[self.widget_id]["page_sizes"]

    def set_page_sizes(self, value: List[int]):
        self._page_sizes = value
        DataJson()[self.widget_id]["page_sizes"] = self._page_sizes
        DataJson().send_changes()

    def current_change(self, func):
        route_path = self.get_route_path(Pagination.Routes.CURRENT_CHANGE)
        server = self._sly_app.get_server()
        self._current_change = True

        @server.post(route_path)
        def _click():
            res = self.get_current_page()
            func(res)

        return _click

    def size_change(self, func):
        route_path = self.get_route_path(Pagination.Routes.SIZE_CHANGE)
        server = self._sly_app.get_server()
        self._size_change = True

        @server.post(route_path)
        def _click():
            res = self.get_page_size()
            func(res)

        return _click
