from typing import List

from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class ContainerWidget(BaseWidget):
    def __init__(self, widgets: List[BaseWidget]):
        super().__init__("container")
        self.widgets = widgets

    def to_html(self) -> str:
        s = "<div>"
        for widget in self.widgets:
            s += "<div>" + widget.to_html() + "</div>"
        s += "</div>"

    def save_data(self, basepath: str) -> None:
        for widget in self.widgets:
            widget.save_data(basepath)

    def save_state(self, basepath: str) -> None:
        for widget in self.widgets:
            widget.save_state(basepath)
