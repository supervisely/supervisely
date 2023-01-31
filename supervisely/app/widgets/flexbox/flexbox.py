from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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
# coding: utf-8


class Flexbox(Widget):
    ALIGN_NAMES = {
        "top": "flex-start",
        "center": "center",
        "bottom": "flex-end",
    }
    # https://www.w3schools.com/css/css3_flexbox.asp
    def __init__(
        self,
        widgets: List[Widget],
        gap: int = 10,
        center_content: bool = False,
        widget_id: str = None,
        vertical_align: Literal["top", "center", "bottom"] = None,
    ):
        self._widgets = widgets
        self._gap = gap
        self._center_content = center_content
        if vertical_align is not None and vertical_align in self.ALIGN_NAMES.keys():
            self._vertical_align = self.ALIGN_NAMES[vertical_align]
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self) -> Dict:
        res = {
            "center": self._center_content,
            "verticalAlign": self._vertical_align,
        }
        return res

    def get_json_state(self) -> Dict:
        return {}
