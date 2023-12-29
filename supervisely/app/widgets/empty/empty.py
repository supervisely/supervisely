from typing import Optional

from supervisely.app.widgets import Widget


class Empty(Widget):
    """Empty widget is a simple placeholder widget that can be used to create empty spaces within a layout.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/layouts-and-containers/empty>`_
        (including screenshots and examples).

    :param widget_id: Unique widget identifier.
    :type widget_id: str, optional
    :param style: CSS styles to be applied to the widget.
    :type style: str, optional

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import Empty, Text, Container

        empty = Empty()
        text1 = Text('Text 1')
        text2 = Text('Text 2')

        container = Container([text1, empty, text2])
    """

    def __init__(self, widget_id: Optional[str] = None, style: Optional[str] = ""):
        self._style: str = style

        super().__init__(widget_id, file_path=__file__)

    def get_json_data(self) -> None:
        """Empty widget does not have any data and this method always returns None."""
        return None

    def get_json_state(self) -> None:
        """Empty widget does not have any state and this method always returns None."""
        return None
