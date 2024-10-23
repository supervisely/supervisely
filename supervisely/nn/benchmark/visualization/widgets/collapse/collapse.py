from typing import Dict, List

from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


class CollapseWidget(BaseWidget):
    def __init__(self, widgets: List[BaseWidget]) -> None:
        super().__init__()
        self.widgets = widgets

    def save_data(self, basepath: str) -> None:
        for widget in self.widgets:
            widget.save_data(basepath)

    def get_state(self) -> Dict:
        return {}

    def to_html(self) -> str:
        items = "\n".join(
            [
                f"""
                    <el-collapse-item title="{widget.title}">
                        {widget.to_html()}
                    </el-collapse-item>
                """
                for widget in self.widgets
            ]
        )
        return f"""
                    <el-collapse class='mb-6'>
                        {items}
                    </el-collapse>
                """
