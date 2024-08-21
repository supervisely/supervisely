from typing import List


def generate_main_template(metric_visualizations: List):
    template_str = """<div>
    <sly-iw-sidebar :options="{ height: 'calc(100vh - 130px)', clearMainPanelPaddings: true, leftSided: false,  disableResize: true, sidebarWidth: 300 }">
        <div slot="sidebar">"""

    for vis in metric_visualizations:
        template_str += vis.template_sidebar_str

    template_str += """\n        </div>
      
        <div style="padding: 0 15px;">"""

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
            <sly-iw-markdown
              id="{{ widget_id }}"
              class="markdown-no-border"
              iw-widget-id="{{ widget_id }}"
              :actions="{
                'init': {
                  'dataSource': '{{ data_source }}',
                },
              }"
              :command="{{ command }}"
              :data="{{ data }}"
            />
"""

template_chart_str = """
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


template_table_str = """<sly-iw-table
                iw-widget-id="{{ widget_id }}"
                style="cursor: pointer;"
                :options="{ isRowClickable: true }" 
                :actions="{
                  'init': {
                    'dataSource': '{{ init_data_source }}',
                  },
                  'chart-click': {
                    'dataSource': '{{ table_click_data }}',
                    'galleryId': '{{ table_gallery_id }}',                    
                    'getKey':(payload)=>payload.row[0],
                   },
                }"
              :command="{{ command }}"
              :data="{{ data }}"
            />"""

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