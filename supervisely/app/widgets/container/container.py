from typing import List, Literal, Optional

from supervisely.app.widgets import Widget


class Container(Widget):
    """Container widget in Supervisely is a flexible tool that allows for organizing other widgets within it.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/layouts-and-containers/container>`_
        (including screenshots and examples).

    :param widgets: list of widgets to be placed in the container
    :type widgets: Optional[List[Widget]]
    :param direction: direction of the container, one of: vertical, horizontal
    :type direction: Optional[Literal["vertical", "horizontal"]]
    :param gap: gap between widgets in pixels
    :type gap: Optional[int]
    :param fractions: list of fractions for each widget (only for horizontal direction)
    :type fractions: Optional[List[int]]
    :param overflow: overflow behavior, one of: scroll, wrap
    :type overflow: Optional[Literal["scroll", "wrap"]]
    :param style: CSS style for the container
    :type style: Optional[str]
    :param widget_id: An identifier of the widget.
    :type widget_id: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Container, Text

        container = Container(
            widgets=[
                Text("First widget"),
                Text("Second widget"),
            ],
            direction="horizontal",
            gap=10,
            fractions=[1, 2],
            overflow="scroll",
            style="background-color: #f0f0f0; padding: 10px",
        )
    """

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
