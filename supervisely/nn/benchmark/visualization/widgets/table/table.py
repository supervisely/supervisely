import json
from typing import Any, Dict, Optional, Union

from jinja2 import Template

from supervisely.io.fs import ensure_base_path
from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


class TableWidget(BaseWidget):
    def __init__(
        self,
        name: str,
        data: Dict = None,
        click_data: Any = None,
        click_gallery_id: str = "",
        fix_columns: int = 0,
        show_header_controls: bool = True,
        main_column: Optional[str] = None,
        width: Optional[Union[int, str]] = None,
        page_size: int = 10,
    ) -> None:
        super().__init__(name=name)
        self.data = data
        self.fix_columns = fix_columns
        self.show_header_controls = show_header_controls
        self.main_column = main_column
        self.click_data = click_data
        self.click_gallery_id = click_gallery_id

        if isinstance(width, int):
            width = f"width: {width}px"
        elif isinstance(width, str):
            width = f"width: {width.rstrip('%')}%"
        self.width = width
        self.page_size = page_size

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

    def get_state(self) -> Dict:
        return {}

    def set_click_data(self, click_gallery_id: str, click_data: Any) -> None:
        self.click_handled = True
        self.click_data = click_data
        self.clickable = True
        self.click_gallery_id = click_gallery_id

    def get_render_data(self) -> Dict:
        return {
            "widget_id": self.id,
            "fixColumns": self.fix_columns,
            "showHeaderControls": self.show_header_controls,
            "init_data_source": self.data_source,
            "table_click_data": self.click_data_source,
            "table_gallery_id": self.click_gallery_id,
            "mainColumn": self.main_column,
            "clickable": self.clickable,
            "width": self.width or "",
            "data": "data",
            "command": "command",
            "page_size": self.page_size,
        }

    def to_html(self) -> str:
        template_str = """
            <div style="margin-top: 20px; margin-bottom: 30px; {{ width }}">
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
                        'perPage': {{ page_size }},
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
                    slot-scope='{ row, column, cellValue }'
                    >
                    <div
                        v-if="column === '{{ mainColumn }}'"
                        class="fflex"
                    >
                        <b>{{ '{{cellValue}}' }}</b>
                    </div>
                    </span>
                </sly-iw-table>
            </div>
        """
        return Template(template_str).render(self.get_render_data())
