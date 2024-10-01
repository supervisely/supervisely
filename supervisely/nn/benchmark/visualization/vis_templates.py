from typing import List


def generate_main_template(metric_visualizations: List):
    template_str = """<div class="model-benchmark-body">

    <sly-style>
    .model-benchmark-body .sly-markdown-widget .markdown-body { padding: 0; font-family: inherit; }
    .model-benchmark-body .sly-markdown-widget .markdown-body h2 { font-size: 18px; font-weight: 600; margin-bottom: 0px; border: 0; }
    .model-benchmark-body .sly-markdown-widget .markdown-body h3 { color: #949bab; font-size: 18px; margin-bottom: 7px; }
    .model-benchmark-body .sly-markdown-widget .markdown-body p { margin-bottom: 12px; }
    .model-benchmark-body .el-collapse { margin: 15px 0; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1); border-radius: 7px; width: fit-content; }
    .model-benchmark-body .el-collapse .el-collapse-item__header { background: transparent; padding-right: 15px; }
    .model-benchmark-body .model-info-block { display: flex; gap: 10px; align-items: center; margin: 0 0 15px; color: #778592; font-size: 13px; }
    .model-benchmark-body .model-info-block > div { display: flex; gap: 4px; align-items: center; }
    /* , .model-benchmark-body .sly-markdown-widget .markdown-body>*:last-child */
    .model-benchmark-body .sly-iw-notification-box .notification-box.notification-box-info { width: fit-content; }
    .model-benchmark-body h1 { font-size: 20px; font-weight: bold; margin-bottom: 5px; }
    .model-benchmark-body .overview-info-block { background: #f4f7fb; width: fit-content; border-radius: 12px; padding: 16px; margin-bottom: 20px; }
    .model-benchmark-body .overview-info-block ul { list-style: none; padding: 0; }
    .model-benchmark-body .overview-info-block ul p { padding: 0; }
    .model-benchmark-body .sly-sidebar-widget .main-wrapper .sidebar-panel { top: 10px; }

    </sly-style>

    <sly-iw-sidebar
        :options="{ height: 'calc(100vh - 130px)', clearMainPanelPaddings: true, leftSided: false,  disableResize: true, sidebarWidth: 300 }"
    >
        <div slot="sidebar">"""

    for vis in metric_visualizations:
        template_str += vis.template_sidebar_str

    template_str += """\n        </div>

        <div style="padding-right: 35px;">"""

    for vis in metric_visualizations:
        template_str += """\n                <div style="margin-top: 20px;">"""
        template_str += vis.template_main_str
        template_str += """\n                </div>"""

    template_str += "\n        </div>\n    </sly-iw-sidebar>"

    template_str += """\n
        <sly-iw-gallery
            ref='modal_general'
            iw-widget-id='modal_general'
            :options="{'isModalWindow': true}"
            :actions="{
                'init': {
                'dataSource': '/data/modal_general.json',
                },
            }"
            :command="command"
            :data="data"
        /> \n
        <sly-iw-gallery
            ref='modal_general_diff'
            iw-widget-id='modal_general_diff'
            :options="{'isModalWindow': true}"
            :actions="{
                'init': {
                'dataSource': '/data/modal_general_diff.json',
                },
                'chart-click': {
                    'dataSource': '/data/gallery_explorer_grid_diff_data.json',
                    'galleryId': 'modal_general',
                        'limit': 3,
                },
            }"
            :command="command"
            :data="data"
        > \n
            <span slot="image-left-header">
                <i class="zmdi zmdi-collection-image"></i> Compare with GT
            </span>
        </sly-iw-gallery>
        </div>"""

    return template_str


template_markdown_str = """
            <div style="margin-top: 10px;">
                <sly-iw-markdown
                id="{{ widget_id }}"
                class="markdown-no-border {{ 'overview-info-block' if is_overview else '' }}"
                iw-widget-id="{{ widget_id }}"
                :actions="{
                    'init': {
                    'dataSource': '{{ data_source }}',
                    },
                }"
                :command="{{ command }}"
                :data="{{ data }}"
                />
            </div>
"""

template_chart_str = """
            <div style="margin-top: 20px; margin-bottom: 20px;">
                <sly-iw-chart
                iw-widget-id="{{ widget_id }}"{% if switchable %}
                v-show="state.{{ radio_group }} === '{{ switch_key }}'"
                {% endif %}:actions="{
                    'init': {
                    'dataSource': '{{ init_data_source }}',
                    },{% if chart_click_data_source %}
                    'chart-click': {
                    'dataSource': '{{ chart_click_data_source }}',{% if cls_name in ['outcome_counts'] %}
                    'getKey': (payload) => payload.points[0].data.name,{% endif %}{% if cls_name in ['frequently_confused', 'recall', 'precision', 'recall_vs_precision'] %}
                    'getKey': (payload) => payload.points[0].label,{% endif %}{% if cls_name in ['pr_curve_by_class'] %}
                    'getKey': (payload) => payload.points[0].data.legendgroup,{% endif %}{% if cls_name in ['per_class_avg_precision'] %}
                    'getKey': (payload) => payload.points[0].theta,{% endif %}{% if cls_name in ['per_class_outcome_counts'] %}
                    'getKey': (payload) => `${payload.points[0].label}${'-'}${payload.points[0].data.name}`,{% endif %}{% if cls_name in ['confusion_matrix', 'per_class_outcome_counts'] %}
                    'keySeparator': '{{ key_separator }}',{% endif %}
                    'galleryId': 'modal_general',
                    'limit': 9
                    },{% endif %}
                }"
                :command="{{ command }}"
                :data="{{ data }}"
                />
            </div>


"""

template_radiogroup_str = """<el-radio v-model="state.{{ radio_group }}" label="{{ switch_key }}" style="margin-top: 10px;">{{ switch_key }}</el-radio>"""


template_gallery_str = """<sly-iw-gallery
            iw-widget-id="{{ widget_id }}"
            {% if is_table_gallery %}
            ref='{{ widget_id }}'
            {% endif %}
            :actions="{
                'init': {
                'dataSource': '{{ init_data_source }}',
                },
                {% if gallery_diff_data_source %}
                'chart-click': {
                    'dataSource': '{{ gallery_diff_data_source }}',
                    'getKey':(payload)=>payload['annotation']['image_id'],
                    'galleryId': 'modal_general',
                        'limit': 3,
                },
                {% endif %}
            }"
            :command="{{ command }}"
            :data="{{ data }}"
            >
                {% if gallery_diff_data_source %}
                    <span slot="image-left-header">
                        <i class="zmdi zmdi-collection-image"></i> Compare with GT
                    </span>
                {% endif %}
            </sly-iw-gallery>

              {% if gallery_click_data_source %}
              <div style="display: flex; justify-content: center; margin-top:10px;" >                
                <el-button iw-widget-id="btn-1" type="primary" @click="command({
                  method: 'update-gallery',
                  payload: {
                    data: {
                      'key': 'explore',
                      'limit': 9,
                      'dataSource': '{{ gallery_click_data_source }}',
                    },
                    'galleryId': 'modal_general_diff',
                  },
                  internalCommand: true
                })">Explore all predictions</el-button>
                </div> {% endif %}
"""


template_table_str = """
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

template_notification_str = """
            <div style="margin-top: 20px; margin-bottom: 20px;">
                <sly-iw-notification              
                iw-widget-id="{{ widget_id }}"
                :data="{{ data }}"
                >
                <span slot="title">
                    {{ title }}
                </span>

                <span slot="description">
                    {{ description }}
                </span>
                </sly-iw-notification>
            </div>"""
