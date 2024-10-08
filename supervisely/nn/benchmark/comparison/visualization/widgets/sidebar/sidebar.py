from typing import List

from jinja2 import Template

from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class SidebarWidget(BaseWidget):

    def __init__(self, widgets: List[BaseWidget], anchors: List[str]) -> None:
        self.widgets = widgets
        self.anchors = anchors

    def sidebar_template_str(self, widget: BaseWidget):
        button_style = f"{{fontWeight: data.scrollIntoView === '{widget.id}' ? 'bold' : 'normal'}}"

        return f"""
            <div>
                <el-button type="text" @click="data.scrollIntoView='{widget.id}'" :style="{button_style}">
                {widget.title}
                </el-button>
            </div>
        """

    @property
    def html_str(self):
        anchored_widgets = sorted(
            [widget for widget in self.widgets if widget.id in self.anchors],
            key=lambda x: self.anchors.index(x.id),
        )
        sidebar_options = """
        <sly-iw-sidebar
            :options="{ height: 'calc(100vh - 130px)', clearMainPanelPaddings: true, leftSided: false, disableResize: true, sidebarWidth: 300 }"
        >
            <div slot="sidebar">
        """
        sidebar_content = "".join(self.sidebar_template_str(widget) for widget in anchored_widgets)
        main_content = "".join(
            f"""
            <div style="margin-top: 20px;">
                {widget.to_html()}
            </div>
            """
            for widget in self.widgets
        )
        closing_tags = "\n    </div>\n</sly-iw-sidebar>"

        return sidebar_options + sidebar_content + main_content + closing_tags

    def to_html(self) -> str:
        Template(self.html_str).render()

    def save_data(self, path: str) -> None:
        for widget in self.widgets:
            widget.save_data(path)

    def save_state(self, path: str) -> None:
        for widget in self.widgets:
            widget.save_state(path)
