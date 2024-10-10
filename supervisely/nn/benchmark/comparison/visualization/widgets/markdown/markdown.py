import json

from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


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

    def save_data(self, basepath: str) -> None:
        # init data
        basepath = basepath.rstrip("/")
        ensure_base_path(basepath + self.data_source)

        with open(basepath + self.data_source, "w") as f:
            json.dump(self.text, f)

    def save_state(self, basepath: str) -> None:
        return

    def to_html(self) -> str:
        is_overview = self.title == "Overview"
        return f"""
            <div style="margin-top: 10px;">
                <sly-iw-markdown
                id="{ self.id }"
                class="markdown-no-border { 'overview-info-block' if is_overview else '' }"
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
