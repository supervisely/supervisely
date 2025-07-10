from typing import List, Tuple, Union

from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
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
        widgth: int = 370,
        widget_id: str = None,
        file_path: str = __file__,
        copy_annotations: bool = True,
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
        self.widgth = widgth
        self._copy_annotations = copy_annotations
        self.project_info = (
            self._api.project.get_info_by_id(self.project_id) if self.project_id else None
        )
        self.all_datasets = (
            self._api.dataset.get_list(self.project_id, recursive=True) if self.project_id else []
        )
        self.items_count = self._count_items(self.all_datasets, datasets_ids, with_children=True)
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
        self.nested_datasets_checkbox = Checkbox(
            "Include nested datasets",
            checked=True,
        )
        self.input_project_container = Container(
            widgets=[
                self.input_datasets_select,
                self.nested_datasets_checkbox,
                self.project_preview,
            ],
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
            Text("Copy annotations from source project", font_size=13), checked=self._copy_annotations
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

    def _datasets_to_process(
        self, all_datasets: List[DatasetInfo], datasets_ids: List[int], with_children: bool
    ) -> List[DatasetInfo]:
        if datasets_ids is None:
            return all_datasets.copy()
        datasets = []
        for ds in [ds for ds in all_datasets if ds.id in datasets_ids]:
            datasets.append(ds)
            if with_children:
                children = [child for child in all_datasets if child.parent_id == ds.id]
                if children:
                    datasets.extend(
                        self._datasets_to_process(
                            all_datasets, [child.id for child in children], True
                        )
                    )
        return datasets

    def _count_items(
        self, all_datasets: List[DatasetInfo], datasets_ids: List[int], with_children: bool
    ) -> int:
        return sum(
            ds.items_count
            for ds in self._datasets_to_process(
                all_datasets, datasets_ids, with_children=with_children
            )
        )

    def _selected_text(self, datasets_ids: List[int] = None, with_children: bool = True) -> str:
        red = "#FF6458;"
        blue = "rgb(0, 154, 255);"
        datasets = self._datasets_to_process(
            self.all_datasets, datasets_ids, with_children=with_children
        )
        color = blue if self.items_count > 0 else red
        ds_num = len(datasets)
        return f'Selected <b style="color: {color};">{ds_num}</b> dataset{"s" if ds_num % 10 != 1 else ""} with <b style="color: {color};">{self.items_count}</b> videos'

    def _update_preview(self):
        with_children = self.nested_datasets_checkbox.is_checked()
        self.selected_items_text.text = self._selected_text(
            self.datasets_ids, with_children=with_children
        )
        if self.items_count > 0:
            self.run_button.enable()
        else:
            self.run_button.disable()

    def _datasets_changed(self, datasets_ids: List[int]):
        self.preview_container.loading = True
        self.datasets_ids = datasets_ids
        project_id = self.input_datasets_select.get_selected_project_id()
        with_children = self.nested_datasets_checkbox.is_checked()
        if self.project_id != project_id:
            if project_id is None:
                self.project_id = None
                self.project_info = None
                self.all_datasets = []
            else:
                self.project_id = project_id
                self._api.project.get_info_by_id(project_id)
                self.all_datasets = self._api.dataset.get_list(project_id, recursive=True)
        self.items_count = self._count_items(
            self.all_datasets, datasets_ids, with_children=with_children
        )
        self._update_preview()
        self.preview_container.loading = False

    def _init_peview_gui(self):
        self.selected_items_text = Text("", font_size=13)
        self.run_button = Button("Run", icon="zmdi zmdi-play", button_size="mini")
        self.run_button.disable()

        self.preview_text = Text(
            '<i class="zmdi zmdi-eye" style="padding-right: 10px; color: rgb(0, 154, 255);"></i><b style="font-size: 14px">Preview</b>',
            font_size=13,
        )
        self.preview_field = Container(
            widgets=[
                self.preview_text,
                Container(widgets=[self.selected_items_text], style="padding-left: 21px;"),
            ],
            style="padding-top: 10px;",
        )
        self.run_button_container = Container(
            widgets=[self.run_button],
            direction="horizontal",
            overflow="wrap",
            style="display: flex; justify-content: flex-end;",
            widgets_style="display: flex; flex: none;",
        )
        self.preview_container = Container(
            widgets=[self.preview_field, self.run_button_container],
        )
        self._update_preview()

        @self.input_datasets_select.value_changed
        def on_input_datasets_select_change(datasets_ids: List[int]):
            self._datasets_changed(datasets_ids)

        @self.nested_datasets_checkbox.value_changed
        def on_nested_datasets_checkbox_change(is_checked: bool):
            self._datasets_changed(self.input_datasets_select.get_selected_ids())

        @self.run_button.click
        def on_run_button_click():
            self.run()

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
        if self.output_project_id is not None:
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
        self._init_peview_gui()
        self._init_progress_gui()

        self.content = Container(
            widgets=[
                self.input_field,
                self.settings_field,
                self.output_field,
                self.preview_container,
                self.result_container,
            ],
            style=f"width: {self.widgth}px;",
        )

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
            with_children = self.nested_datasets_checkbox.is_checked()
            selected_datasets_with_children = self._datasets_to_process(
                all_datasets=self.all_datasets,
                datasets_ids=datasets_ids,
                with_children=with_children,
            )
            total_items = sum(ds.items_count for ds in selected_datasets_with_children)
            datasets_ids = [ds.id for ds in selected_datasets_with_children]
            project_info = self.project_info

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
                dst_project_info = sample_video_project(
                    api=self._api,
                    project_id=project_id,
                    settings=self._get_settings(),
                    dst_project_id=self._get_dst_project_id(),
                    datasets_ids=datasets_ids,
                    items_progress_cb=pbar.update,
                    video_progress=frames_pbar,
                )
        except Exception as e:
            self.error_notification.set(title="Error", description=str(e))
            self.error_notification.show()
            raise e
        else:
            if self.output_project_selectable:
                dst_project_info = self._api.project.get_info_by_id(dst_project_info.id)
                self.result_project_preview.set(dst_project_info)
                self.result_project_preview_field.show()
            else:
                dst_project_info = self._api.project.get_info_by_id(self.output_project_id)
                self.output_project_preview.set(dst_project_info)
        finally:
            self.progress_container.hide()

    @property
    def selected_project_id(self) -> int:
        if not self.project_selectable:
            return self.project_id
        return self.input_datasets_select.get_selected_project_id()

    @selected_project_id.setter
    def selected_project_id(self, value: int):
        if not self.project_selectable:
            raise ValueError("Project is not selectable.")
        self.input_datasets_select.set_project_id(value)
        self._datasets_changed(self.input_datasets_select.get_selected_ids())

    @property
    def selected_all_datasets(self) -> bool:
        return self.input_datasets_select._all_datasets_checkbox.is_checked()

    @selected_all_datasets.setter
    def selected_all_datasets(self, value: bool):
        if value:
            self.input_datasets_select._all_datasets_checkbox.check()
        else:
            self.input_datasets_select._all_datasets_checkbox.uncheck()
        self._datasets_changed(self.input_datasets_select.get_selected_ids())

    @property
    def selected_datasets_ids(self) -> List[int]:
        return self.input_datasets_select.get_selected_ids()

    @selected_datasets_ids.setter
    def selected_datasets_ids(self, value: List[int]):
        self.input_datasets_select.set_dataset_ids(value)

    @property
    def include_nested_datasets(self) -> bool:
        return self.nested_datasets_checkbox.is_checked()

    @include_nested_datasets.setter
    def include_nested_datasets(self, value: bool):
        if value:
            self.nested_datasets_checkbox.check()
        else:
            self.nested_datasets_checkbox.uncheck()
        self._datasets_changed(self.input_datasets_select.get_selected_ids())

    @property
    def step(self) -> int:
        return self.step_input.get_value()

    @step.setter
    def step(self, value: int):
        self.step_input.value = value

    @property
    def only_annotated(self) -> bool:
        return self.only_annotated_checkbox.is_checked()

    @only_annotated.setter
    def only_annotated(self, value: bool):
        if value:
            self.only_annotated_checkbox.check()
        else:
            self.only_annotated_checkbox.uncheck()

    @property
    def resize(self) -> Union[Tuple[int, int], None]:
        if self.resize_checkbox.is_checked():
            return (self.resize_input_h.get_value(), self.resize_input_w.get_value())
        return None

    @resize.setter
    def resize(self, value: Union[Tuple[int, int], None]):
        if value is None:
            self.resize_checkbox.uncheck()
        else:
            self.resize_checkbox.check()
            self.resize_input_h.value = value[0]
            self.resize_input_w.value = value[1]

    @property
    def copy_annotations(self) -> bool:
        return self.copy_annotations_checkbox.is_checked()

    @copy_annotations.setter
    def copy_annotations(self, value: bool):
        if value:
            self.copy_annotations_checkbox.check()
        else:
            self.copy_annotations_checkbox.uncheck()

    @property
    def selected_output_project_id(self) -> int:
        if not self.output_project_selectable:
            return self.output_project_id
        if self.output_mode_radio.get_value() == "create":
            return None
        return self.output_project_select.get_selected_id()

    @selected_output_project_id.setter
    def selected_output_project_id(self, value: int):
        if not self.output_project_selectable:
            raise ValueError("Output project is not selectable.")
        self.output_project_select.set_project_id(value)
        if value is not None:
            self.output_mode_radio.set_value("merge")
        else:
            self.output_mode_radio.set_value("create")

    def get_json_data(self) -> dict:
        return {}

    def get_json_state(self) -> dict:
        return {}

    def to_html(self):
        return self.content.to_html()
