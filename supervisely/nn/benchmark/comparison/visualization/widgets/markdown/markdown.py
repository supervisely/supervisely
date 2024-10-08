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

    def save_data(self, basepath: str) -> None:
        return

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
                        :command="command"
                        :data="{ self.text }"
                        />
                    </div>
        """
