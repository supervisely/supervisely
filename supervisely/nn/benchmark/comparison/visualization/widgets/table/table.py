from typing import List, Tuple, Union, Optional

from supervisely.nn.benchmark.comparison.visualization.widgets.widget import BaseWidget


class TableWidget(BaseWidget):
    def __init__(
        self,
        data_source: str,
        click_data: str = None,
        clickable: bool = False,
        fix_columns: int = 0,
        show_header_controls: bool = True,
        main_column: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.clickable = clickable
        self.fix_columns = fix_columns
        self.show_header_controls = show_header_controls
        self.data_source = data_source
        self.click_data = click_data
        self.main_column = main_column

    def save_data(self, basepath: str) -> None:
        return

    def save_state(self, basepath: str) -> None:
        return

    def to_html(self) -> str:
        return f"""
                    <div style="margin-top: 20px; margin-bottom: 30px;">
                        <sly-iw-table
                            iw-widget-id="{ self.id }"
                            {% if self.clickable %}
                                style="cursor: pointer;"
                            {% endif %}
                            :options="{
                                isRowClickable: '{ self.clickable }' === 'True',
                                fixColumns: { self.fix_columns }, 
                                showHeaderControls: '{ self.show_header_controls }' === 'True',
                            }"
                            :actions="{
                            'init': {
                                'dataSource': '{ self.data_source }',
                            },
                            {% if self.clickable %}
                            'chart-click': {
                                'dataSource': '{ self.click_data }',
                                'galleryId': 'modal_general',
                                'getKey':(payload)=>payload.row[0],
                            },
                            {% endif %}
                            }"
                        :command="command"
                        :data="{ self.data_source }"
                        >
                            <span
                            slot="custom-cell-content"
                            slot-scope="{ row, column, cellValue }"
                            >
                            <div
                                v-if="column === '{ self.main_column }'"
                                class="fflex"
                            >
                                <b>{{ '{{ cellValue }}' }}</b>
                            </div>
                            </span>
                        </sly-iw-table>
                    </div>
        """
