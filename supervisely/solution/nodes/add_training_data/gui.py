from typing import Callable, List, Tuple

from supervisely import logger
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    Checkbox,
    CheckboxField,
    Container,
    Dialog,
    Flexbox,
    NotificationBox,
    ProjectDatasetTable,
    ReloadableArea,
    StepperProgress,
    Text,
    Widget,
)
from supervisely.nn.training.gui.train_val_splits_selector import (
    TrainValSplits,
    TrainValSplitsSelector,
)
from supervisely.project.project import ProjectType


# At the moment it is very tricky to get resulting DatasetInfos as they are being created externally - Not implemented yet.
class AddTrainingDataGUI(Widget):
    """
    GUI for Add Training Data node.
    Allows users to configure random split settings.
    """

    def __init__(self, api: Api, *args, **kwargs):
        """
        Initialize the Add Training Data node.
        """
        self.api = api
        self._on_settings_saved_callbacks: List[Callable] = []
        self.widget = self._create_main_widget()
        super().__init__(*args, **kwargs)

    @property
    def modal(self) -> Dialog:
        if not hasattr(self, "_modal"):
            self._modal = Dialog(
                title="Add Training Data",
                content=self.widget,
            )
        return self._modal

    @property
    def project_table(self) -> ProjectDatasetTable:
        if not hasattr(self, "_project_table"):
            self._project_table = ProjectDatasetTable(
                page_size=8, allowed_project_types=[ProjectType.IMAGES]
            )
        return self._project_table

    @property
    def replicate_structure_checkbox(self) -> CheckboxField:
        if not hasattr(self, "_replicate_structure_checkbox"):
            self._replicate_structure_checkbox = CheckboxField(
                title="Replicate Dataset Structure",
                description="If checked, resulting nested datasets will feature the same structure as in the source project.",
                remove_margins=True,
            )
            self._replicate_structure_checkbox.hide()
        return self._replicate_structure_checkbox

    @property
    def stepper(self) -> StepperProgress:
        if not hasattr(self, "_stepper"):
            steps = [
                StepperProgress.StepItem("Select Project"),
                StepperProgress.StepItem("Select Datasets"),
                StepperProgress.StepItem("Configure Splits"),
                StepperProgress.StepItem("Add Data"),
            ]
            self._stepper = StepperProgress(steps)
        return self._stepper

    @property
    def select_all_datasets_checkbox(self) -> CheckboxField:
        if not hasattr(self, "_select_all_datasets_checkbox"):
            self._select_all_datasets_checkbox = CheckboxField(
                title="Select All Datasets",
                description="Select all datasets in the currently selected project.",
                checked=True,
                remove_margins=True,
            )
            self._select_all_datasets_checkbox.hide()

            @self._select_all_datasets_checkbox.value_changed
            def on_select_all_datasets_checkbox_change(is_checked):
                if not self.project_table.current_table == self.project_table.CurrentTable.DATASETS:
                    return
                if is_checked:
                    self.project_table.table.select_rows(
                        [i for i in range(self.project_table.table._rows_total)]
                    )
                    self.project_table.disable()
                else:
                    self.project_table.enable()
                    self.project_table.table.select_rows([])

        return self._select_all_datasets_checkbox

    @property
    def splits_widget(self) -> TrainValSplitsSelector:
        if not hasattr(self, "_splits_widget"):
            # self._splits_widget = TrainValSplits(
            #     # project_id=project_id,
            #     random_splits=True,
            #     tags_splits=True,
            #     datasets_splits=True,
            #     collections_splits=True,
            # )
            # self._splits_widget.hide()
            self._splits_widget = TrainValSplitsSelector(self.api)
            self._splits_widget.train_val_splits.hide()

        return self._splits_widget

    def _create_main_widget(self) -> Container:
        desc = Text(
            "Skip the initial steps of data import and annotation "
            "by copying already labeled data from your existing projects to "
            "Training Project. Select a team, workspace, and project — and the "
            "app will automatically detect and split datasets using naming "
            "conventions (train/val) or your defined Train/Val Split settings. "
            "All selected data will be copied into your training project "
            "and organized into collections for immediate use."
        )

        btns_container = self._init_btns()

        table_container = Container(
            [
                Container(
                    [
                        self.select_all_datasets_checkbox,
                        self.project_table,
                        self.splits_widget.train_val_splits,
                    ],
                    gap=0,
                ),
                self.replicate_structure_checkbox,
            ]
        )

        return Container(
            [
                desc,
                self.stepper,
                table_container,
                btns_container,
            ],
            gap=20,
        )

    def _init_btns(self) -> Flexbox:
        next_btn = Button(
            "Next",
            button_size="small",
            icon="zmdi zmdi-arrow-right",
            style="primary",
        )
        next_btn.disable()
        back_btn = Button("Back", plain=True, icon="zmdi zmdi-arrow-left", button_size="small")
        back_btn.hide()

        @self.project_table.table.selection_changed
        def on_table_selection_change(selected_items):
            if selected_items:
                next_btn.enable()
            else:
                next_btn.disable()

        @self.splits_widget.train_val_splits.value_changed
        def on_splits_value_changed(val):
            if val:
                next_btn.enable()
            else:
                next_btn.disable()

        @next_btn.click
        def on_next_btn_click():
            if self.project_table.current_table == self.project_table.CurrentTable.PROJECTS:
                self.project_table.switch_table(self.project_table.CurrentTable.DATASETS)
                self.stepper.next_step()
                if self.project_table.has_nested_datasets():
                    self.replicate_structure_checkbox.show()
                self.select_all_datasets_checkbox.show()
                if self.select_all_datasets_checkbox.is_checked():
                    self.project_table.disable()
                    self.project_table.table.select_rows(
                        [i for i in range(self.project_table.table._rows_total)]
                    )
                    next_btn.enable()
                back_btn.show()
            elif (
                self.project_table.current_table == self.project_table.CurrentTable.DATASETS
                and self.splits_widget.train_val_splits.is_hidden()
            ):
                self.stepper.next_step()
                self._set_train_val_splits_data()
                self.replicate_structure_checkbox.hide()
                self.select_all_datasets_checkbox.hide()
                next_btn.text = "Add"
                next_btn.enable()
            elif (
                self.project_table.current_table == self.project_table.CurrentTable.DATASETS
                and not self.splits_widget.train_val_splits.is_hidden()
            ):
                self.stepper.next_step()
                self.modal.hide()
                self._trigger_settings_saved_event()
                # area to reset the widget to initial state
                next_btn.text = "Next"
                next_btn.disable()
                back_btn.hide()
                self.project_table.show()
                self.splits_widget.train_val_splits.hide()
                self.stepper.set_active_step(1)
                self.project_table.switch_table(self.project_table.CurrentTable.PROJECTS)
                self.project_table.table.clear_selection()

        @back_btn.click
        def on_back_btn_click():
            self.project_table.show()
            self.stepper.previous_step()
            self.splits_widget.train_val_splits.hide()
            if (
                self.project_table.current_table == self.project_table.CurrentTable.DATASETS
                and not self.splits_widget.train_val_splits.is_hidden()
            ):
                self.replicate_structure_checkbox.hide()
                self.select_all_datasets_checkbox.hide()
                back_btn.hide()
                self.project_table.switch_table(self.project_table.CurrentTable.PROJECTS)
                next_btn.text = "Next"
                self.project_table.table.clear_selection()
                next_btn.disable()
            elif (
                self.project_table.current_table == self.project_table.CurrentTable.DATASETS
                and self.splits_widget.train_val_splits.is_hidden()
            ):
                self.project_table.show()
                next_btn.text = "Next"
                next_btn.enable()
                self.project_table.switch_table(self.project_table.CurrentTable.DATASETS)
                if self.project_table.has_nested_datasets():
                    self.replicate_structure_checkbox.show()
                self.select_all_datasets_checkbox.show()
                if self.select_all_datasets_checkbox.is_checked():
                    self.project_table.disable()
                    self.project_table.table.select_rows(
                        [i for i in range(self.project_table.table._rows_total)]
                    )

        return Flexbox(
            [Container([back_btn, next_btn], direction="horizontal")],
            horizontal_alignment="flex-end",
        )

    def _set_train_val_splits_data(self) -> None:
        # TODO: determine whether to include tabs & check which tab we need to select by default
        self.project_table.hide()
        self.splits_widget.train_val_splits.show()
        project_id = self.get_selected_project_id()
        if not project_id:
            raise RuntimeError("Project ID is not selected. Cannot set splits data.")
        dataset_ids = self.get_selected_dataset_ids()
        self.splits_widget.set_project_id(project_id, dataset_ids)
        self.splits_widget._detect_splits(True, True, True)

    def get_json_data(self) -> dict:
        return {}

    def get_json_state(self) -> dict:
        return {}

    def get_selected_project_id(self):
        return self.project_table.get_selected_project_id()

    def get_selected_dataset_ids(self):
        return self.project_table.get_selected_dataset_ids()

    def on_settings_saved(self, callback: Callable[[dict], None]) -> Callable:
        self._on_settings_saved_callbacks.append(callback)
        return callback

    def _trigger_settings_saved_event(self):
        """Trigger all registered settings saved callbacks"""
        selected_ids_to_parents = {}
        ds_infos = sorted(self.project_table.get_selected_datasets(), key=lambda x: x.id)
        full_names = self.project_table.get_selected_datasets_full_names()
        for ds_info, full_name in zip(ds_infos, full_names):
            ds_name = ds_info.name
            parents = full_name.removesuffix(ds_name).rstrip("/")
            selected_ids_to_parents[ds_info.id] = parents.split("/")

        split_method = self.splits_widget.get_split_method()
        split_iteminfos = self.splits_widget.get_splits()
        train_split_ids, val_split_ids = self._get_ids_from_iteminfos(
            split_method, *split_iteminfos
        )

        settings_data = {
            "workspace_id": self.project_table.team_workspace_selector.get_selected_workspace_id(),
            "team_id": self.project_table.team_workspace_selector.get_selected_team_id(),
            "project_id": self.get_selected_project_id(),
            "dataset_ids": self.get_selected_dataset_ids(),
            "splits_ids": (train_split_ids, val_split_ids),
            "splits_item": split_iteminfos,
            "replicate_structure": self.replicate_structure_checkbox.is_checked(),
            "selected_ids_to_parents": selected_ids_to_parents,
        }

        for callback in self._on_settings_saved_callbacks:
            try:
                callback(settings_data)
            except Exception as e:
                logger.error(f"Error in settings saved callback: {e}")

    def _get_ids_from_iteminfos(
        self, split_method, train_iteminfos, val_iteminfos
    ) -> Tuple[List[int], List[int]]:
        if split_method in ["Based on collections", "Based on item tags"]:
            project_id = self.project_table.get_selected_project_id()
            dataset_infos = self.api.dataset.get_list(project_id, recursive=True)
            ds_name_to_id = {ds_info.name: ds_info.id for ds_info in dataset_infos}
        else:
            ds_name_to_id = {
                ds_info.name: ds_info.id for ds_info in self.project_table.get_selected_datasets()
            }

        train_split, val_split = [], []
        for split, split_list in ((train_iteminfos, train_split), (val_iteminfos, val_split)):
            split_iteminfos = [iteminfo for iteminfo in split]
            split_ids = [iteminfo.id for iteminfo in split_iteminfos]
            split_datasets = set()
            for iteminfo in split_iteminfos:
                split_datasets.add(iteminfo.dataset_name)

            for dataset_name in split_datasets:
                dataset_id = ds_name_to_id.get(dataset_name)
                if dataset_id is None:
                    raise RuntimeError(
                        f"Dataset '{dataset_name}' from train split is not in the selected datasets."
                    )

                filters = [{"field": "id", "operator": "in", "value": split_ids}]
                image_infos = self.api.image.get_list(dataset_id, filters=filters)
                image_ids = [image_info.id for image_info in image_infos]
                split_list.extend(image_ids)

        return train_split, val_split
