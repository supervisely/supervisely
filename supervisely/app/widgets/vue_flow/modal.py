from typing import List, Optional, Union

from supervisely.app.widgets import (
    Container,
    Dialog,
    ReloadableArea,
    Widget,
    generate_id,
)


class VueFlowModal(Dialog):
    def __init__(
        self,
        widget_id_prefix: Optional[str] = None,
        widget_id: Optional[str] = None,
    ):
        if widget_id_prefix:
            widget_id = generate_id(widget_id_prefix)
        elif widget_id:
            widget_id = widget_id
        self.reloadable_area = ReloadableArea()
        super().__init__(
            title="",
            content=self.reloadable_area,
            widget_id=widget_id,
            size="small",
        )

    def set_content(self, content: Union[Widget, List[Widget]]) -> None:
        if isinstance(content, list):
            content = Container(widgets=content)
        self.reloadable_area.set_content(content)
        self.reloadable_area.reload()

    def set_content(self, content: Union[Widget, List[Widget]]) -> None:
        if isinstance(content, list):
            content = Container(widgets=content)
        self.reloadable_area.set_content(content)
        self.reloadable_area.reload()
