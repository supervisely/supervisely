from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Widget

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Carousel(Widget):
    """Carousel is a widget in Supervisely that allows loop a series of images or texts in a limited space on the UI.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/media/carousel>`_
        (including screenshots and examples).

    :param items: List of Carousel.Item objects to be displayed in the carousel.
    :type items: List[Carousel.Item]
    :param height: Height of the carousel in pixels.
    :type height: Optional[int]
    :param initial_index: Index of the item to be displayed at initialization.
    :type initial_index: Optional[int]
    :param trigger: Trigger type to change the active item.
    :type trigger: Optional[Literal["hover", "click"]]
    :param autoplay: If True, the carousel will be autoplayed.
    :type autoplay: Optional[bool]
    :param interval: Time interval between each autoplay.
    :type interval: Optional[int]
    :param indicator_position: Position of the indicator.
    :type indicator_position: Optional[Literal["outside", "none"]]
    :param arrow: Arrow display type.
    :type arrow: Optional[Literal["always", "hover", "never"]]
    :param type: Carousel type.
    :type type: Optional[Literal["card"]]
    :param widget_id: Unique widget identifier.
    :type widget_id: str

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Carousel

        carousel_items = [
            Carousel.Item(name="item1", label="Item 1"),
            Carousel.Item(name="item2", label="Item 2"),
            Carousel.Item(name="item3", label="Item 3"),
        ]

        carousel = Carousel(
            items=carousel_items, height=350, initial_index=0, trigger="click",
            autoplay=False, interval=3000, indicator_position="none", arrow="hover", type=None
            )
    """

    class Routes:
        VALUE_CHANGED = "value_changed"

    class Item:
        """Represents an item in the carousel.

        :param name: Name of the item.
        :type name: Optional[str]
        :param label: Label of the item.
        :type label: Optional[str]
        :param is_link: If True, the item will be displayed as a link.
        :type is_link: Optional[bool]
        """

        def __init__(
            self,
            name: Optional[str] = "",
            label: Optional[str] = "",
            is_link: Optional[bool] = True,
        ):
            self.name = name
            self.label = label
            self.is_link = is_link

        def to_json(self) -> Dict[str, Union[str, bool]]:
            """Returns dictionary with item data.

            Dictionary contains the following fields:
                - name: Name of the item.
                - label: Label of the item.
                - is_link: If True, the item will be displayed as a link.
            """
            return {
                "name": self.name,
                "label": self.label,
                "is_link": self.is_link,
            }

    def __init__(
        self,
        items: List[Carousel.Item],
        height: Optional[int] = 350,
        initial_index: Optional[int] = 0,
        trigger: Optional[Literal["hover", "click"]] = "click",
        autoplay: Optional[bool] = False,
        interval: Optional[int] = 3000,
        indicator_position: Optional[Literal["outside", "none"]] = "none",
        arrow: Optional[Literal["always", "hover", "never"]] = "hover",
        type: Optional[Literal["card"]] = None,
        widget_id: Optional[str] = None,
    ):
        self._height = f"{height}px"
        self._items = items
        self._initial_index = initial_index
        self._trigger = trigger
        self._autoplay = autoplay
        self._interval = interval
        self._indicator_position = indicator_position if indicator_position is not None else "none"
        self._arrow = arrow
        self._type = type

        self._changes_handled = False
        self._clicked_value = None

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _set_items(self):
        return [item.to_json() for item in self._items]

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - height: Height of the carousel in pixels.
            - items: List of Carousel.Item objects to be displayed in the carousel.
            - initialIndex: Index of the item to be displayed at initialization.
            - trigger: Trigger type to change the active item.
            - autoplay: If True, the carousel will be autoplayed.
            - interval: Time interval between each autoplay.
            - indicatorPosition: Position of the indicator.
            - arrow: Arrow display type.
            - type: Carousel type.
        """

        return {
            "height": self._height,
            "items": self._set_items(),
            "initialIndex": self._initial_index,
            "trigger": self._trigger,
            "autoplay": self._autoplay,
            "interval": self._interval,
            "indicatorPosition": self._indicator_position,
            "arrow": self._arrow,
            "type": self._type,
        }

    def get_json_state(self) -> Dict[str, Any]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - clickedValue: Name of the item that was clicked.

        :return: Dictionary with widget state.
        :rtype: Dict[str, Any]
        """
        return {"clickedValue": self._clicked_value}

    def get_active_item(self) -> int:
        """Returns index of the active item.

        :return: Index of the active item.
        :rtype: int
        """
        return StateJson()[self.widget_id]["clickedValue"]

    def get_items(self) -> List[Carousel.Item]:
        """Returns list of Carousel.Item objects.

        :return: List of Carousel.Item objects.
        :rtype: List[Carousel.Item]
        """
        return DataJson()[self.widget_id]["items"]

    def set_items(self, value: List[Carousel.Item]) -> None:
        """Sets list of Carousel.Item objects to be displayed in the carousel.
        This method will overwrite the existing items, not append to it.
        To append items, use :meth:`add_items`.

        :param value: List of Carousel.Item objects to be displayed in the carousel.
        :type value: List[Carousel.Item]
        :raises ValueError: If items are not of type Carousel.Item.
        """
        if not all(isinstance(item, Carousel.Item) for item in value):
            raise ValueError("Items must be of type Carousel.Item")
        self._items = value
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def add_items(self, value: List[Carousel.Item]) -> None:
        """Appends list of Carousel.Item objects to the existing items.
        This method will not overwrite the existing items, but append to it.
        To overwrite items, use :meth:`set_items`.

        :param value: List of Carousel.Item objects to be displayed in the carousel.
        :type value: List[Carousel.Item]
        """
        self._items.extend(value)
        DataJson()[self.widget_id]["items"] = self._set_items()
        DataJson().send_changes()

    def set_height(self, value: int) -> None:
        """Sets height of the carousel.

        :param value: Height of the carousel in pixels.
        :type value: int
        :raises ValueError: If height value is not an integer.
        """
        if type(value) is not int:
            raise ValueError("Height value must be an integer")
        self._height = f"{value}px"
        DataJson()[self.widget_id]["height"] = self._height
        DataJson().send_changes()

    def get_initial_index(self) -> int:
        """Returns index of the item to be displayed at initialization.

        :return: Index of the item to be displayed at initialization.
        :rtype: int
        """
        return DataJson()[self.widget_id]["initialIndex"]

    def set_initial_index(self, value: int) -> None:
        """Sets index of the item to be displayed at initialization.

        :param value: Index of the item to be displayed at initialization.
        :type value: int
        :raises ValueError: If index value exceeds the size of the carousel items.
        """
        if value >= len(self._items):
            raise ValueError("Index of the value being set exceeds the size of the carousel items.")
        self._initial_index = value
        DataJson()[self.widget_id]["initialIndex"] = self._initial_index
        DataJson().send_changes()

    def value_changed(self, func: Callable[[str], Any]) -> Callable[[], None]:
        """Decorator for the function to be called when the value of the carousel is changed.

        :param func: Function to be called when the value of the carousel is changed.
        :type func: Callable[[str], Any]
        :return: Decorated function.
        :rtype: Callable[[], None]
        """
        route_path = self.get_route_path(Carousel.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            curr_idx = self.get_active_item()
            curr_name = self._items[curr_idx].name
            func(curr_name)

        return _click
