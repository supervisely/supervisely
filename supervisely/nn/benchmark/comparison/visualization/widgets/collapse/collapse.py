from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class CollapseWidget(BaseWidget):
    def __init__(self, widgets: list) -> None:
        super().__init__()
        self.widgets = widgets

    def save_data(self, basepath: str) -> None:
        return

    def save_state(self, basepath: str) -> None:
        return

    def to_html(self) -> str:
        return f"""
                    <el-collapse class='mb-6'>
                        {
                            "".join([
                                f"""
                                    <el-collapse-item title="{subwidget.title}">
                                        {{ {subwidget.name}_html }}
                                    </el-collapse-item>
                                """
                                for subwidget in self.widgets
                            ])
                        }
                    </el-collapse>
        """