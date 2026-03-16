from typing import Any, Callable, Dict, List, Optional

from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets


class ReorderTable(Widget):
    """Table widget that lets users reorder rows via drag-and-drop, multi-select,
    and a floating action panel (Top / Up / Set to # / Down / Bottom).
    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/tables/reordertable>`_.

    :Usage example:

        .. code-block:: python

            from supervisely.app.widgets import ReorderTable

            table = ReorderTable(
                columns=["Name", "Score"],
                data=[["Alice", 95], ["Bob", 87], ["Carol", 92]],
                page_size=10,
            )

            @table.order_changed
            def handle_reorder(order: list):
                print("New order:", order)
    """

    class Routes:
        """Callback route names used by the widget frontend to notify Python."""

        ORDER_CHANGED = "order_changed_cb"

    def __init__(
        self,
        columns: List[str],
        data: Optional[List[List]] = None,
        page_size: int = 10,
        content_top_right: Optional[Widget] = None,
        widget_id: Optional[str] = None,
    ):
        """
        :param columns: Column header names.
        :type columns: List[str]
        :param data: Row data; each inner list is one row of cell values.
        :type data: List[List], optional
        :param page_size: Number of rows per page.
        :type page_size: int, optional
        :param content_top_right: Widget to render in the top-right corner of the table header.
        :type content_top_right: Widget, optional
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        """
        if data is None:
            data = []

        self._columns = list(columns)
        self._data = [list(row) for row in data]
        self._validate_data(self._columns, self._data)
        self._page_size = max(1, int(page_size))
        self._order: List[int] = list(range(len(self._data)))
        self._changes_handled = False
        self._content_top_right = content_top_right

        super().__init__(widget_id=widget_id, file_path=__file__)

        script_path = "./sly/css/app/widgets/reorder_table/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

    @staticmethod
    def _validate_data(columns: List[str], data: List[List]) -> None:
        expected = len(columns)
        for idx, row in enumerate(data):
            if len(row) != expected:
                raise ValueError(
                    f"Row {idx} has {len(row)} cells but expected {expected} (number of columns)"
                )

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data.

        :returns: Dictionary with ``columns``, ``rows`` and ``total`` fields.
        :rtype: Dict[str, Any]
        """
        return {
            "columns": self._columns,
            "rows": self._data,
            "total": len(self._data),
        }

    def get_json_state(self) -> Dict[str, Any]:
        """Returns dictionary with widget state.

        :returns: Dictionary with ``order``, ``page``, ``pageSize`` and ``selectedPositions`` fields.
        :rtype: Dict[str, Any]
        """
        return {
            "order": list(self._order),
            "page": 1,
            "pageSize": self._page_size,
            "selectedPositions": [],
        }

    def get_order(self) -> List[int]:
        """Returns the current row order as a list of 0-based original row indices.

        :returns: Current order; position *i* holds the original 0-based row index.
        :rtype: List[int]
        """
        self._order = list(StateJson()[self.widget_id].get("order", self._order))
        return list(self._order)

    def get_reordered_data(self) -> List[List]:
        """Returns all rows arranged in the current user-defined order.

        :returns: Reordered rows.
        :rtype: List[List]
        """
        self._order = list(StateJson()[self.widget_id].get("order", self._order))
        return [self._data[i] for i in self._order]

    def get_column_data(self, column_name: str) -> List[Any]:
        """Returns values from a single column in the current row order.

        :param column_name: Name of the column to retrieve.
        :type column_name: str
        :returns: List of cell values for the given column in the current order.
        :rtype: List[Any]
        :raises ValueError: If ``column_name`` is not found in the table columns.
        """
        if column_name not in self._columns:
            raise ValueError(
                f"Column '{column_name}' not found. Available columns: {self._columns}"
            )
        self._order = list(StateJson()[self.widget_id].get("order", self._order))
        col_idx = self._columns.index(column_name)
        return [self._data[i][col_idx] for i in self._order]

    def set_data(self, columns: List[str], data: List[List]) -> None:
        """Replaces the table contents and resets the order to the identity permutation.

        :param columns: New column header names.
        :type columns: List[str]
        :param data: New row data.
        :type data: List[List]
        """
        self._columns = list(columns)
        self._data = [list(row) for row in data]
        self._validate_data(self._columns, self._data)
        self._order = list(range(len(self._data)))

        DataJson()[self.widget_id].update(
            {
                "columns": self._columns,
                "rows": self._data,
                "total": len(self._data),
            }
        )
        DataJson().send_changes()

        StateJson()[self.widget_id].update(
            {
                "order": list(self._order),
                "page": 1,
                "selectedPositions": [],
            }
        )
        StateJson().send_changes()

    def reset_order(self) -> None:
        """Resets the row order to the original (identity) permutation."""
        self._order = list(range(len(self._data)))
        StateJson()[self.widget_id].update(
            {
                "order": list(self._order),
                "page": 1,
                "selectedPositions": [],
            }
        )
        StateJson().send_changes()

    def is_order_changed(self) -> bool:
        """Returns whether the current row order differs from the original order.

        :returns: ``True`` if rows were reordered, otherwise ``False``.
        :rtype: bool
        """
        return self.get_order() != list(range(len(self._data)))


    def order_changed(self, func: Callable[[List[int]], None]) -> Callable:
        """Decorator that registers a callback invoked whenever the row order changes.

        The callback receives one argument: the new order as a list of 0-based
        original row indices (same value returned by :py:meth:`get_order`).

        :param func: Callback function accepting a single ``List[int]`` argument.
        :type func: Callable[[List[int]], None]
        :returns: The original function unchanged.
        :rtype: Callable

        :Usage example:

            .. code-block:: python

                @table.order_changed
                def on_reorder(order):
                    print("Reordered:", order)
        """
        route_path = self.get_route_path(ReorderTable.Routes.ORDER_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _order_changed_handler():
            self._order = list(StateJson()[self.widget_id].get("order", self._order))
            func(list(self._order))

        return func
