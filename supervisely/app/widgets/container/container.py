from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from supervisely.app import DataJson
from supervisely.app.widgets import Widget


class Container(Widget):
    def __init__(
        self,
        widgets: List[Widget] = [],
        direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 10,
        fractions: List[int] = None,
        # overflow: Literal["scroll", "wrap"] = None,
        # grid_cell_width: Literal["20%", "300px"] = None,
        widget_id: str = None,
    ):
        self._widgets = widgets
        self._direction = direction
        self._gap = gap
        self._overflow = "scroll"
        self._grid_cell_width = None  # grid_cell_width
        if self._direction == "vertical" and fractions is not None:
            raise ValueError("fractions can be defined only with horizontal direction")

        if fractions is not None and len(widgets) != len(fractions):
            raise ValueError(
                "len(widgets) != len(fractions): fractions have to be defined for all widgets"
            )

        if self._direction == "vertical":
            self._overflow = None
            if self._overflow is not None:
                raise ValueError(
                    "overflow argument can only be defined if direction is 'horizontal'"
                )
            if self._grid_cell_width is not None:
                raise ValueError(
                    "grid_cell_width argument can only be defined if direction is 'horizontal'"
                )

        if self._direction == "horizontal" and self._overflow is None:
            self._overflow = "wrap"

        if self._grid_cell_width is not None and self._overflow != "wrap":
            raise ValueError(
                "grid_cell_width argument can only be defined if overflow is 'wrap'"
            )

        self._fractions = fractions
        self._flex_direction = "column"
        if direction == "horizontal":
            self._flex_direction = "row"
            if self._fractions is None:
                if self._grid_cell_width is None:
                    self._fractions = ["1 1 auto"] * len(self._widgets)
                else:
                    self._fractions = [
                        f"1 1 calc({self._grid_cell_width} - {self._gap}px)"
                    ] * len(self._widgets)
                # self._fractions = ["1"] * len(self._widgets)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None
