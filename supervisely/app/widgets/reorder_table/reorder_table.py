from typing import Any, Callable, Dict, List, Optional

from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.app.widgets_context import JinjaWidgets


class ReorderTable(Widget):
    """Table widget that lets users reorder rows via drag-and-drop, multi-select,
    and action buttons (top / up / set-to-position / down / bottom).

    Features
    --------
    * Any number of custom columns supplied as plain strings.
    * Data supplied as a list of lists (one inner list per row).
    * Server-side pagination with an in-widget editable page-size control.
    * Each row shows its **original** 1-based position badge; if the item has
      been moved a second badge shows the **current** position.
    * Multi-row selection via checkboxes (including select-all on the current
      page).
    * Drag-and-drop reordering within the visible page.
    * Floating action panel that appears when at least one row is selected:
      ``Top``, ``Up``, ``Set to # <input>``, ``Down``, ``Bottom``, ``Deselect``.

    Usage Example
    -------------
    .. code-block:: python

        from supervisely.app.widgets import ReorderTable

        table = ReorderTable(
            columns=["Name", "Score"],
            data=[["Alice", 95], ["Bob", 87], ["Carol", 92]],
            page_size=10,
        )

        @table.order_changed
        def handle_reorder(order: list):
            # order is a list of 0-based original row indices in the new order
            print("New order:", order)
    """

    class Routes:
        ORDER_CHANGED = "order_changed_cb"

    def __init__(
        self,
        columns: List[str],
        data: Optional[List[List]] = None,
        page_size: int = 10,
        widget_id: Optional[str] = None,
    ):
        """
        :param columns: Column header names shown in the table.
        :type columns: List[str]
        :param data: Row data; each element is a list of cell values (one per column).
            Defaults to an empty table.
        :type data: List[List], optional
        :param page_size: Initial number of rows displayed per page.
        :type page_size: int, optional
        :param widget_id: Unique widget identifier (auto-generated if omitted).
        :type widget_id: str, optional
        """
        if data is None:
            data = []

        self._columns = list(columns)
        self._data = [list(row) for row in data]
        self._page_size = max(1, int(page_size))
        self._order: List[int] = list(range(len(self._data)))
        self._order_changed_handled = False
        self._order_changed_cb: Optional[Callable[[List[int]], None]] = None

        super().__init__(widget_id=widget_id, file_path=__file__)

        # Register the Vue component script (de-duplicated by class name).
        script_path = "./sly/css/app/widgets/reorder_table/script.js"
        JinjaWidgets().context["__widget_scripts__"][self.__class__.__name__] = script_path

        # Register backend route.
        route_path = self.get_route_path(ReorderTable.Routes.ORDER_CHANGED)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def _order_changed_handler():
            self._order = list(StateJson()[self.widget_id].get("order", self._order))
            if self._order_changed_handled and self._order_changed_cb is not None:
                self._order_changed_cb(list(self._order))

    # ------------------------------------------------------------------ #
    #  Widget data / state                                                 #
    # ------------------------------------------------------------------ #

    def get_json_data(self) -> Dict[str, Any]:
        return {
            "columns": self._columns,
            "rows": self._data,
            "total": len(self._data),
        }

    def get_json_state(self) -> Dict[str, Any]:
        return {
            "order": list(self._order),
            "page": 1,
            "pageSize": self._page_size,
            "selectedPositions": [],
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_order(self) -> List[int]:
        """Return the current row order as a list of 0-based original row indices.

        :returns: Current order list; index *i* contains the original 0-based
            row index that is now at position *i*.
        :rtype: List[int]
        """
        return list(self._order)

    def get_reordered_data(self) -> List[List]:
        """Return all data rows arranged in the current user-defined order.

        :returns: Reordered rows.
        :rtype: List[List]
        """
        return [self._data[i] for i in self._order]

    def set_data(self, columns: List[str], data: List[List]) -> None:
        """Replace the table contents and reset the order to the identity permutation.

        Broadcasts changes to all connected frontends immediately.

        :param columns: New column header names.
        :type columns: List[str]
        :param data: New row data.
        :type data: List[List]
        """
        self._columns = list(columns)
        self._data = [list(row) for row in data]
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
        """Reset the row order to the original (identity) permutation.

        Broadcasts changes to all connected frontends immediately.
        """
        self._order = list(range(len(self._data)))
        StateJson()[self.widget_id].update(
            {
                "order": list(self._order),
                "page": 1,
                "selectedPositions": [],
            }
        )
        StateJson().send_changes()

    # ------------------------------------------------------------------ #
    #  Event decorator                                                     #
    # ------------------------------------------------------------------ #

    def order_changed(self, func: Callable[[List[int]], None]) -> Callable:
        """Decorator that registers a callback invoked whenever the row order changes.

        The callback receives one argument: the new order as a list of 0-based
        original row indices (same value as :py:meth:`get_order`).

        .. code-block:: python

            @table.order_changed
            def on_reorder(order):
                print("Reordered:", order)

        :param func: Callback function.
        :type func: Callable[[List[int]], None]
        :returns: The original function (unchanged).
        :rtype: Callable
        """
        self._order_changed_handled = True
        self._order_changed_cb = func
        return func
