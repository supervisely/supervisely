from typing import Any
from supervisely.app.widgets import Widget


class Draggable(Widget):
    """Wrapper widget that makes its content draggable within the UI."""

    def __init__(self, content: Widget, key: Any = None, widget_id: str = None):
        """Initialize Draggable.

        :param content: Widget to make draggable.
        :type content: :class:`~supervisely.app.widgets.widget.Widget`
        :param key: Optional key for drag state. Defaults to content.widget_id.
        :type key: Any, optional
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        """
        self._content = content
        self._key = key
        if key is None:
            self._key = content.widget_id
        super().__init__(widget_id=widget_id, file_path=__file__)
    
    def get_json_data(self):
        return {}
    
    def get_json_state(self):
        return {}
