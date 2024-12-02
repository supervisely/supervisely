from typing import Dict

from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


class NotificationWidget(BaseWidget):

    def __init__(
        self,
        name: str,
        title: str,
        desc: str = None,
    ) -> None:
        super().__init__(name, title=title)
        self.desc = desc

    def save_data(self, basepath: str) -> None:
        return

    def get_state(self) -> Dict:
        return {}

    def to_html(self) -> str:
        res = f"""
                    <div style="margin-top: 20px; margin-bottom: 20px;">
                        <sly-iw-notification              
                        iw-widget-id="{ self.id }"
                        >"""
        if self.title:
            res += f"""
                            <span slot="title">
                                { self.title }
                            </span>"""
        if self.desc:
            res += f"""
                            <span slot="description">
                                { self.desc }
                            </span>"""
        res += """
                        </sly-iw-notification>
                    </div>"""
        return res
