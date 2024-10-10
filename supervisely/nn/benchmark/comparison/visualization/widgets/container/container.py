from typing import Dict, List

from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class ContainerWidget(BaseWidget):
    def __init__(self, widgets: List[BaseWidget], name: str = "container", title: str = None):
        super().__init__(name, title)
        self.widgets = widgets

    def to_html(self) -> str:
        s = "<div>"
        for widget in self.widgets:
            s += "<div>" + widget.to_html() + "</div>"
        s += "</div>"
        return s

    def save_data(self, basepath: str) -> None:
        for widget in self.widgets:
            widget.save_data(basepath)

    def get_state(self) -> Dict:
        state = {}
        for widget in self.widgets:
            state.update(widget.get_state())
        return state
