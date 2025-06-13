from typing import Dict, List

from jinja2 import Template

from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


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

        sidebar_content = "".join(self.sidebar_template_str(widget) for widget in anchored_widgets)
        main_content = "".join(
            f"""
            <div style="margin-top: 20px;">
                {widget.to_html()}
            </div>
            """
            for widget in self.widgets
        )
        template = f"""
            <sly-iw-sidebar
                :options="{{ height: 'calc(100vh - 130px)', clearMainPanelPaddings: true, leftSided: false, disableResize: true, sidebarWidth: 300 }}"
            >
                <div slot="sidebar">
                    {sidebar_content}
                </div>
                <div style="padding-right: 35px;">
                    {main_content}
                </div>
            </sly-iw-sidebar>
        """

        return template

    def to_html(self) -> str:
        return Template(self.html_str).render()

    def save_data(self, basepath: str) -> None:
        for widget in self.widgets:
            widget.save_data(basepath)

    def get_state(self) -> Dict:
        state = {}
        for widget in self.widgets:
            state.update(widget.get_state())
        return state
