import json
from typing import Any, Dict, Optional

from jinja2 import Template

from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class TableWidget(BaseWidget):
    def __init__(
        self,
        data: Dict = None,
        click_data: Any = None,
        click_gellery_id: str = "",
        fix_columns: int = 0,
        show_header_controls: bool = True,
        main_column: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.fix_columns = fix_columns
        self.show_header_controls = show_header_controls
        self.main_column = main_column
        self.click_data = click_data
        self.click_gellery_id = click_gellery_id

        self.clickable = self.click_data is not None

    def save_data(self, basepath: str) -> None:
        # init data
        basepath = basepath.rstrip("/")
        ensure_base_path(basepath + self.data_source)

        with open(basepath + self.data_source, "w") as f:
            json.dump(self.data, f)

        # click data
        if self.click_data is not None:
            ensure_base_path(basepath + self.click_data_source)
            with open(basepath + self.click_data_source, "w") as f:
                json.dump(self.click_data, f)

    def save_state(self, basepath: str) -> None:
        return

    def get_render_data(self) -> Dict:
        return {
            "widget_id": self.id,
            "fixColumns": self.fix_columns,
            "showHeaderControls": self.show_header_controls,
            "init_data_source": self.data_source,
            "table_click_data": self.click_data_source,
            "table_gallery_id": self.click_gellery_id,
            "mainColumn": self.main_column,
            "clickable": self.clickable,
            "data": "data",
            "command": "command",
        }

    def to_html(self) -> str:
        template_str = """
            <div style="margin-top: 20px; margin-bottom: 30px;">
                <sly-iw-table
                    iw-widget-id="{{ widget_id }}"
                    {% if clickable %}
                        style="cursor: pointer;"
                    {% endif %}
                    :options="{
                        isRowClickable: '{{ clickable }}' === 'True',
                        fixColumns: {{ fixColumns }}, 
                        showHeaderControls: '{{ showHeaderControls }}' === 'True',
                    }"
                    :actions="{
                    'init': {
                        'dataSource': '{{ init_data_source }}',
                    },
                        {% if clickable %}
                    'chart-click': {
                        'dataSource': '{{ table_click_data }}',
                        'galleryId': '{{ table_gallery_id }}',
                        'getKey':(payload)=>payload.row[0],
                    },
                        {% endif %}
                    }"
                :command="{{ command }}"
                :data="{{ data }}"
                >
                    <span
                    slot="custom-cell-content"
                    slot-scope="{ row, column, cellValue }"
                    >
                    <div
                        v-if="column === '{{ mainColumn }}'"
                        class="fflex"
                    >
                        <b>Batch size {{ '{{ cellValue }}' }}</b>
                    </div>
                    </span>
                </sly-iw-table>
            </div>
        """
        Template(template_str).render(self.get_render_data())
