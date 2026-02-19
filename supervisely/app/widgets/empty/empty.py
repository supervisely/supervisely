from typing import Optional

from supervisely.app.widgets import Widget


class Empty(Widget):
    """Placeholder for empty space in layout."""

    def __init__(self, widget_id: Optional[str] = None, style: Optional[str] = ""):
        """
        :param widget_id: Widget identifier.
        :param style: CSS styles.

        :Usage Example:

            .. code-block:: python

                from supervisely.app.widgets import Empty, Container, Text
                container = Container([Text("A"), Empty(), Text("B")])
        """
        self._style: str = style

        super().__init__(widget_id, file_path=__file__)

    def get_json_data(self) -> None:
        """Empty widget does not have any data and this method always returns None."""
        return None

    def get_json_state(self) -> None:
        """Empty widget does not have any state and this method always returns None."""
        return None
