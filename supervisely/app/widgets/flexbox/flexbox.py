from typing import Dict, List, Literal

from supervisely.app.widgets import Widget

"""
<div 
    {% if not loop.last %}
    style="margin-bottom: {{{widget._gap}}}px;"
    {% endif %}
>
    {{{w}}}
</div>
"""


class Flexbox(Widget):
    """Flexbox layout widget for arranging child widgets vertically with configurable gap and alignment."""

    def __init__(
        self,
        widgets: List[Widget],
        gap: int = 10,
        center_content: bool = False,
        widget_id: str = None,
        vertical_alignment: Literal["start", "end", "center", "stretch", "baseline"] = None,
    ):
        """
        :param widgets: List of child widgets.
        :type widgets: List[:class:`~supervisely.app.widgets.widget.Widget`]
        :param gap: Vertical gap between widgets in pixels.
        :type gap: int
        :param center_content: If True, center content horizontally.
        :type center_content: bool
        :param widget_id: Unique widget identifier.
        :type widget_id: str, optional
        :param vertical_alignment: Vertical alignment of items.
        :type vertical_alignment: Literal["start", "end", "center", "stretch", "baseline"], optional
        """
        self._widgets = widgets
        self._gap = gap
        self._center_content = center_content
        self._vertical_alignment = vertical_alignment
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {"center": self._center_content}
        return res

    def get_json_state(self) -> Dict:
        return {}
