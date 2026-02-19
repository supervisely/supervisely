from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Container(Widget):
    """Layout container for arranging child widgets vertically or horizontally with gap and fractions."""

    def __init__(
        self,
        widgets: List[Widget] = [],
        direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 10,
        fractions: List[int] = None,
        widget_id: str = None,
    ):
        """Initialize Container (legacy).

        :param widgets: List of child widgets.
        :type widgets: List[Widget]
        :param direction: "vertical" or "horizontal".
        :type direction: Literal["vertical", "horizontal"]
        :param gap: Gap between widgets in pixels.
        :type gap: int
        :param fractions: Width fractions per widget (horizontal only).
        :type fractions: List[int], optional
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional

        :raises ValueError: If fractions used with vertical or len mismatch.
        """
        self._widgets = widgets
        self._direction = direction
        self._gap = gap
        if self._direction == "vertical" and fractions is not None:
            raise ValueError("fractions can be defined only with horizontal direction")

        if fractions is not None and len(widgets) != len(fractions):
            raise ValueError(
                "len(widgets) != len(fractions): fractions have to be defined for all widgets"
            )
        self._fractions = fractions
        self._flex_direction = "column"
        if direction == "horizontal":
            self._flex_direction = "row"
            if self._fractions is None:
                self._fractions = [1] * len(self._widgets)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None
