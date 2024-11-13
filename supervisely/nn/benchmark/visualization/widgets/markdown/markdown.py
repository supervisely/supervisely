from typing import Dict

from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


class MarkdownWidget(BaseWidget):
    def __init__(
        self,
        name: str,
        title: str,
        text: str = None,
    ) -> None:
        super().__init__(name, title)
        self.text = text
        self.data_source = f"/data/{self.name}_{self.id}.md"
        self.is_info_block = False
        self.width_fit_content = False

    def save_data(self, basepath: str) -> None:
        # init data
        basepath = basepath.rstrip("/")
        ensure_base_path(basepath + self.data_source)

        with open(basepath + self.data_source, "w") as f:
            f.write(self.text)

    def get_state(self) -> Dict:
        return {}

    def to_html(self) -> str:
        style_class = "markdown-no-border"
        if self.is_info_block:
            style_class += " overview-info-block"
        if self.width_fit_content:
            style_class += " width-fit-content"

        return f"""
            <div style="margin-top: 10px;">
                <sly-iw-markdown
                id="{ self.id }"
                class="{ style_class }"
                iw-widget-id="{ self.id }"
                :actions="{{
                    'init': {{
                    'dataSource': '{ self.data_source }',
                    }},
                }}"
                :command="command"
                :data="data"
                />
            </div>
        """
