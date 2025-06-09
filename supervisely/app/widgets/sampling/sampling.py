from supervisely.api.api import Api
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.empty.empty import Empty
from supervisely.app.widgets.field.field import Field
from supervisely.app.widgets.flexbox.flexbox import Flexbox
from supervisely.app.widgets.input_number.input_number import InputNumber
from supervisely.app.widgets.one_of.one_of import OneOf
from supervisely.app.widgets.project_thumbnail.project_thumbnail import ProjectThumbnail
from supervisely.app.widgets.radio_group.radio_group import RadioGroup
from supervisely.app.widgets.select_project.select_project import SelectProject
from supervisely.app.widgets.text.text import Text
from supervisely.app.widgets.widget import Widget
from supervisely.project.project import ProjectType
from supervisely.video.sampling import SamplingSettings, sample_video_project


class Sampling(Widget):
    def __init__(
        self,
        project_id: int = None,
        project_selectable: bool = True,
        widget_id: str = None,
        file_path: str = None,
    ):
        super().__init__(widget_id, file_path)
        if not project_selectable and project_id is None:
            raise ValueError(
                "Either 'project_id' must be provided or 'project_selectable' must be True."
            )
        self._api = Api()
        self.project_id = project_id
        self.project_selectable = project_selectable
        self._init_gui()

    def _init_gui(self):
        self.project_select = SelectProject(
            default_id=self.project_id, allowed_types=[ProjectType.VIDEOS], size="mini"
        )
        self.project_preview = ProjectThumbnail()
        self.project_preview.hide()
        if not self.project_selectable:
            self.project_select.hide()
            project_info = self._api.project.get_info_by_id(self.project_id)
            self.project_preview.set(project_info)
            self.project_preview.show()
        self.input_project_container = Container(
            widgets=[self.project_select, self.project_preview]
        )
        self.input_field = Field(
            content=self.input_project_container,
            title="📹 Input project",
        )

        self.settings_icon = Text("⏺︎")
        self.only_annotated_checkbox = Checkbox("Only annotated frames")
        self.only_annotated_row = Flexbox(
            widgets=[self.settings_icon, self.only_annotated_checkbox]
        )
        self.step_label = Text("Step:")
        self.step_input = InputNumber(value=1, min=1, step=1)
        self.step_row = Flexbox(
            widgets=[self.settings_icon, self.step_label, self.step_input],
            vertical_alignment="center",
        )
        self.resize_checkbox = Checkbox("Resize frames")
        self.resize_input_h_label = Text("Height:")
        self.resize_input_h = InputNumber(value=224, min=1, step=1, size="mini", controls=False)
        self.resize_input_w_label = Text("Width:")
        self.resize_input_w = InputNumber(value=224, min=1, step=1, size="mini", controls=False)
        self.resize_h_row = Flexbox(widgets=[self.resize_input_h_label, self.resize_input_h])
        self.resize_w_row = Flexbox(widgets=[self.resize_input_w_label, self.resize_input_w])
        self.resize_row = Flexbox(widgets=[self.settings_icon, self.resize_checkbox])
        self.resize_hw_container = Container(
            widgets=[Empty(), self.resize_h_row, self.resize_w_row], style="padding-left: 17px;"
        )
        self.resize_hw_container.hide()
        self.resize_container = Container(
            widgets=[self.resize_row, self.resize_hw_container], gap=0
        )
        self.copy_annotations_checkbox = Checkbox("Copy annotations from source project")
        self.copy_annotations_row = Flexbox(
            widgets=[self.settings_icon, self.copy_annotations_checkbox]
        )
        self.settings_container = Container(
            widgets=[
                Empty(),
                self.only_annotated_row,
                self.step_row,
                self.resize_container,
                self.copy_annotations_row,
            ],
            style="padding-left: 10px;",
        )
        self.settings_field = Field(
            content=self.settings_container,
            title="⚙️ Sampling settings",
        )

        self.output_project_select = SelectProject(allowed_types=[ProjectType.IMAGES], size="mini")
        self.output_mode_radio = RadioGroup(
            items=[
                RadioGroup.Item("create", "Create new project", content=Empty()),
                RadioGroup.Item(
                    "merge", "Merge with existing project", content=self.output_project_select
                ),
            ]
        )
        self.output_oneof = OneOf(self.output_mode_radio)
        self.output_container = Container(
            widgets=[self.output_mode_radio, self.output_oneof],
        )
        self.output_field = Field(content=self.output_container, title="Output")

        self.run_button = Button("Run sampling", button_size="small", icon="play")
        self.run_button_container = Container

        self.container = Container(
            widgets=[
                self.input_field,
                self.settings_field,
                self.output_field,
                self.run_button,
            ]
        )

        @self.resize_checkbox.value_changed
        def on_resize_checkbox_change(checked: bool):
            if checked:
                self.resize_hw_container.show()
            else:
                self.resize_hw_container.hide()

        @self.run_button.click
        def on_run_button_click():
            self.run()

    def _get_settings(self) -> dict:
        settings = {
            SamplingSettings.ONLY_ANNOTATED: self.only_annotated_checkbox.is_checked(),
            SamplingSettings.STEP: self.step_input.get_value(),
            SamplingSettings.RESIZE: None,
            SamplingSettings.COPY_ANNOTATIONS: self.copy_annotations_checkbox.is_checked(),
        }
        if self.resize_checkbox.is_checked():
            settings[SamplingSettings.RESIZE] = [
                self.resize_input_h.get_value(),
                self.resize_input_w.get_value(),
            ]

        return settings

    def _get_dst_project_id(self) -> int:
        if self.output_mode_radio.get_value() == "create":
            return None
        else:
            return self.output_project_select.get_selected_id()

    def run(self):
        sample_video_project(
            api=self._api,
            project_id=self.project_select.get_selected_id(),
            settings=self._get_settings(),
            dst_project_id=self._get_dst_project_id(),
        )

    def get_json_data(self) -> dict:
        return {}

    def get_json_state(self) -> dict:
        return {}

    def to_html(self):
        return self.container.to_html()
