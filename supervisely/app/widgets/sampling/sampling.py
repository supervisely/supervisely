from typing import List

from supervisely.api.api import Api
from supervisely.app.widgets import Progress
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.empty.empty import Empty
from supervisely.app.widgets.flexbox.flexbox import Flexbox
from supervisely.app.widgets.input_number.input_number import InputNumber
from supervisely.app.widgets.notification_box.notification_box import NotificationBox
from supervisely.app.widgets.one_of.one_of import OneOf
from supervisely.app.widgets.project_thumbnail.project_thumbnail import ProjectThumbnail
from supervisely.app.widgets.radio_group.radio_group import RadioGroup
from supervisely.app.widgets.select_dataset.select_dataset import SelectDataset
from supervisely.app.widgets.select_project.select_project import SelectProject
from supervisely.app.widgets.text.text import Text
from supervisely.app.widgets.widget import Widget
from supervisely.project.project import ProjectType
from supervisely.video.sampling import SamplingSettings, sample_video_project


class Sampling(Widget):
    def __init__(
        self,
        project_id: int = None,
        input_selectable: bool = True,
        datasets_ids: List[int] = None,
        output_project_id: int = None,
        output_project_selectable: bool = True,
        widget_id: str = None,
        file_path: str = None,
    ):
        super().__init__(widget_id, file_path)
        if not input_selectable and project_id is None:
            raise ValueError(
                "Either 'project_id' must be provided or 'project_selectable' must be True."
            )
        if project_id is None and not input_selectable:
            input_selectable = True
        if not output_project_selectable and output_project_id is None:
            raise ValueError(
                "Either 'output_project_id' must be provided or 'output_project_selectable' must be True."
            )
        if output_project_id is None and not output_project_selectable:
            output_project_selectable = True
        self._api = Api()
        self.project_id = project_id
        self.project_selectable = input_selectable
        self.datasets_ids = datasets_ids
        self.output_project_id = output_project_id
        self.output_project_selectable = output_project_selectable
        self._init_gui()

    def _init_input_gui(self):
        self.input_datasets_select = SelectDataset(
            default_id=self.datasets_ids,
            project_id=self.project_id,
            allowed_project_types=[ProjectType.VIDEOS],
            size="mini",
            multiselect=True,
            select_all_datasets=self.datasets_ids is None,
        )
        self.project_preview = ProjectThumbnail()
        self.project_preview.hide()
        if not self.project_selectable:
            self.input_datasets_select.hide()
            project_info = self._api.project.get_info_by_id(self.project_id)
            self.project_preview.set(project_info)
            self.project_preview.show()
        self.input_project_container = Container(
            widgets=[self.input_datasets_select, self.project_preview],
            style="padding-left: 21px; padding-top: 10px;",
        )
        self.input_field = Container(
            widgets=[
                Text(
                    '<i class="zmdi zmdi-collection-video" style="padding-right: 10px; color: rgb(0, 154, 255);"></i><b>Input project</b>'
                ),
                self.input_project_container,
            ],
            gap=0,
        )

    def _init_settings_gui(self):
        self.only_annotated_checkbox = Checkbox(Text("Only annotated frames", font_size=13))
        self.only_annotated_row = Flexbox(widgets=[self.only_annotated_checkbox])
        self.step_label = Text("Step:", font_size=13)
        self.step_input = InputNumber(value=1, min=1, step=1, size="mini", width=160)
        self.step_row = Container(
            widgets=[self.step_label, self.step_input],
            direction="horizontal",
            style="width: 202px; align-items: center;",
            widgets_style="flex: 1; display: flex;",
        )
        self.resize_checkbox = Checkbox(Text("Resize frames", font_size=13))
        self.resize_input_h_label = Text("Height:", font_size=13)
        self.resize_input_h = InputNumber(value=224, min=1, step=1, size="mini", controls=False)
        self.resize_input_w_label = Text("Width:", font_size=13)
        self.resize_input_w = InputNumber(value=224, min=1, step=1, size="mini", controls=False)
        self.resize_h_row = Flexbox(widgets=[self.resize_input_h_label, self.resize_input_h])
        self.resize_w_row = Flexbox(
            widgets=[self.resize_input_w_label, self.resize_input_w], gap=15
        )
        self.resize_row = Flexbox(widgets=[self.resize_checkbox])
        self.resize_hw_container = Container(
            widgets=[Empty(), self.resize_h_row, self.resize_w_row]
        )
        self.resize_hw_container.hide()
        self.resize_container = Container(
            widgets=[self.resize_row, self.resize_hw_container], gap=0
        )
        self.copy_annotations_checkbox = Checkbox(
            Text("Copy annotations from source project", font_size=13)
        )
        self.copy_annotations_row = Flexbox(widgets=[self.copy_annotations_checkbox])

        @self.resize_checkbox.value_changed
        def on_resize_checkbox_change(checked: bool):
            if checked:
                self.resize_hw_container.show()
            else:
                self.resize_hw_container.hide()

        self.settings_container = Container(
            widgets=[
                Empty(),
                self.step_row,
                self.only_annotated_row,
                self.resize_container,
                self.copy_annotations_row,
                Empty(),
            ],
            style="padding-left: 21px; padding-top: 10px;",
        )
        self.settings_field = Container(
            widgets=[
                Text(
                    '<i class="zmdi zmdi-settings" style="padding-right: 10px; color: rgb(0, 154, 255);"></i><b>Sampling settings</b>'
                ),
                self.settings_container,
            ],
            gap=0,
        )

    def _init_output_gui(self):
        self.output_project_select = SelectProject(
            default_id=self.output_project_id, allowed_types=[ProjectType.IMAGES], size="mini"
        )
        self.output_project_preview = ProjectThumbnail()
        self.output_project_preview.hide()
        self.output_mode_radio = RadioGroup(
            items=[
                RadioGroup.Item("create", "Create new project", content=Empty()),
                RadioGroup.Item(
                    "merge",
                    "Merge with existing project",
                    content=self.output_project_select,
                ),
            ]
        )
        self.output_merge_project_oneof = OneOf(self.output_mode_radio)
        self.output_project_container = Container(
            widgets=[self.output_mode_radio, self.output_merge_project_oneof],
        )

        self.output_container = Container(
            widgets=[self.output_project_preview, self.output_project_container],
            gap=0,
            style="padding-left: 21px; padding-top: 10px;",
        )
        if not self.output_project_selectable:
            self.output_project_container.hide()
            project_info = self._api.project.get_info_by_id(self.output_project_id)
            self.output_project_preview.set(project_info)
            self.output_project_preview.show()
        elif self.output_project_id is not None:
            self.output_mode_radio.set_value("merge")

        self.output_field = Container(
            widgets=[
                Text(
                    '<i class="zmdi zmdi-collection-folder-image" style="padding-right: 10px; color: rgb(0, 154, 255);"></i><b>Output project</b>'
                ),
                self.output_container,
            ],
            gap=0,
        )

    def _init_progress_gui(self):
        self.items_progress = Progress(hide_on_finish=False)
        self.frames_progress = Progress(hide_on_finish=False)
        self.error_notification = NotificationBox(title="Error", description="", box_type="error")
        self.error_notification.hide()
        self.progress_container = Container(widgets=[self.items_progress, self.frames_progress])
        self.progress_container.hide()
        self.result_project_preview = ProjectThumbnail()
        self.result_project_preview_field = Container(
            widgets=[
                Text(
                    '<i class="zmdi zmdi-check-square" style="padding-right: 10px; color: rgb(0, 154, 255);"></i><b>Result project</b>'
                ),
                Container(
                    widgets=[self.result_project_preview],
                    style="padding-left: 21px;",
                ),
            ]
        )
        self.result_project_preview_field.hide()
        self.result_container = Container(
            widgets=[
                self.progress_container,
                self.error_notification,
                self.result_project_preview_field,
            ],
            gap=0,
        )

    def _init_gui(self):
        self._init_input_gui()
        self._init_settings_gui()
        self._init_output_gui()
        self._init_progress_gui()

        self.run_button = Button("Run", icon="zmdi zmdi-play", button_size="mini")
        self.run_button_container = Container(
            widgets=[self.run_button],
            direction="horizontal",
            overflow="wrap",
            style="display: flex; justify-content: flex-end;",
            widgets_style="display: flex; flex: none;",
        )

        self.content = Container(
            widgets=[
                self.input_field,
                self.settings_field,
                self.output_field,
                self.run_button_container,
                self.result_container,
            ],
            style="width: 370px;",
        )

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
        self.progress_container.show()
        self.error_notification.hide()
        self.result_project_preview_field.hide()
        try:
            project_id = self.input_datasets_select.get_selected_project_id()
            datasets_ids = self.input_datasets_select.get_selected_ids()
            all_ds_infos = self._api.dataset.get_list(project_id)
            parents = {
                ds.id: ds.parent_id if ds.parent_id is not None else ds.project_id
                for ds in all_ds_infos
            }
            datasets_infos = [ds for ds in all_ds_infos if ds.id in datasets_ids]
            total_items = sum(ds.items_count for ds in datasets_infos)
            project_info = self._api.project.get_info_by_id(project_id)

            if project_info.type != str(ProjectType.VIDEOS):
                raise ValueError(
                    f"Project with ID {self.input_datasets_select.get_selected_id()} is not a video project."
                )
            frames_pbar = self.frames_progress()
            with self.items_progress(
                message=f"Videos progress...",
                total=total_items,
            ) as pbar:
                self.progress_container.show()
                for dataset_info in datasets_infos:
                    sample_video_dataset(
                        api=self._api,
                        dataset_id=dataset_info.id,
                        settings=self._get_settings(),
                        dst_parent_info=parents,
                        items_progress_cb=pbar.update,
                        video_progress=frames_pbar,
                    )
                dst_project_info = self._api.project.get_info_by_id(self._get_dst_project_id())
                dst_project_info = sample_video_project(
                    api=self._api,
                    project_id=project_id,
                    settings=self._get_settings(),
                    dst_project_id=self._get_dst_project_id(),
                    items_progress_cb=pbar.update,
                    video_progress=frames_pbar,
                )
        except Exception as e:
            self.error_notification.set(title="Error", description=str(e))
            self.error_notification.show()
            raise e
        else:
            if self.output_project_selectable:
                self.result_project_preview.set(dst_project_info)
                self.result_project_preview_field.show()
        finally:
            self.progress_container.hide()

    def get_json_data(self) -> dict:
        return {}

    def get_json_state(self) -> dict:
        return {}

    def to_html(self):
        return self.content.to_html()
