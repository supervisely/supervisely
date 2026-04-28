from typing import List, Literal, Optional

from supervisely.app.widgets import Widget


class Container(Widget):
    """Flexible container for organizing child widgets vertically or horizontally."""

    def __init__(
        self,
        widgets: Optional[List[Widget]] = [],
        direction: Optional[Literal["vertical", "horizontal"]] = "vertical",
        gap: Optional[int] = 10,
        fractions: Optional[List[int]] = None,
        overflow: Optional[Literal["scroll", "wrap"]] = "scroll",
        style: Optional[str] = "",
        widget_id: Optional[str] = None,
        widgets_style: Optional[str] = "",
    ):
        """
        :param widgets: List of child widgets.
        :type widgets: Optional[List[Widget]]
        :param direction: "vertical" or "horizontal".
        :type direction: Optional[Literal["vertical", "horizontal"]]
        :param gap: Gap between widgets (px).
        :type gap: Optional[int]
        :param fractions: Width fractions for horizontal layout.
        :type fractions: Optional[List[int]]
        :param overflow: "scroll" or "wrap".
        :type overflow: Optional[Literal["scroll", "wrap"]]
        :param style: CSS for container.
        :type style: Optional[str]
        :param widget_id: Widget identifier.
        :type widget_id: Optional[str]
        :param widgets_style: CSS for child widgets.
        :type widgets_style: Optional[str]

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Container, Text
                container = Container(widgets=[Text("A"), Text("B")], direction="horizontal", gap=10)
        """
        self._widgets = widgets
        self._direction = direction
        self._gap = gap
        self._overflow = overflow
        self._style = style
        self._widgets_style = widgets_style

        if self._overflow not in ["scroll", "wrap", None]:
            raise ValueError("overflow can be only 'scroll', 'wrap' or None")

        if self._direction == "vertical" and self._overflow == "wrap":
            raise ValueError("overflow can be 'wrap' only with horizontal direction")

        if self._direction == "vertical" and fractions is not None:
            raise ValueError("fractions can be defined only with horizontal direction")

        if fractions is not None and len(widgets) != len(fractions):
            raise ValueError(
                "len(widgets) != len(fractions): fractions have to be defined for all widgets"
            )

        if self._direction == "vertical":
            self._overflow = None

        self._fractions = fractions
        self._flex_direction = "column"
        if direction == "horizontal":
            self._flex_direction = "row"
            if self._fractions is None:
                self._fractions = ["1 1 auto"] * len(self._widgets)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> None:
        """The Container widget does not have any data
        the method is overridden to return None."""
        return None

    def get_json_state(self) -> None:
        """The Container widget does not have any state
        the method is overridden to return None."""
        return None
