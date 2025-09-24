from types import MethodType
from typing import Callable, List

from supervisely import logger
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    CheckboxField,
    Container,
    Dialog,
    Flexbox,
    ProjectDatasetTable,
    StepperProgress,
    Text,
    Widget,
)
from supervisely.nn.training.gui.train_val_splits_selector import TrainValSplitsSelector
from supervisely.project.project import ProjectType


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
            self._splits_widget = TrainValSplitsSelector(self.api)
            self._splits_widget.train_val_splits.hide()

            self._splits_widget.train_val_splits.get_splits = MethodType(
                AddTrainingDataGUI._get_splits_downloadless, self._splits_widget.train_val_splits
            )

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
                self.select_all_datasets_checkbox,
                self.project_table,
                self.splits_widget.train_val_splits,
            ],
            gap=0,
        )

        return Container(
            [
                desc,
                Flexbox([self.stepper], center_content=True, horizontal_alignment="center"),
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
            # Step 1 -> Step 2
            if self.project_table.current_table == self.project_table.CurrentTable.PROJECTS:
                self.project_table.switch_table(self.project_table.CurrentTable.DATASETS)
                self.stepper.next_step()
                self.select_all_datasets_checkbox.show()
                if self.select_all_datasets_checkbox.is_checked():
                    self.project_table.disable()
                    self.project_table.table.select_rows(
                        [i for i in range(self.project_table.table._rows_total)]
                    )
                    next_btn.enable()
                back_btn.show()
            # Step 2 -> Step 3
            elif (
                self.project_table.current_table == self.project_table.CurrentTable.DATASETS
                and self.splits_widget.train_val_splits.is_hidden()
            ):
                self.stepper.next_step()
                self.select_all_datasets_checkbox.hide()
                self._set_train_val_splits_data()
                next_btn.text = "Add"
                next_btn.enable()
            # Finish
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
            # Step 2 -> Step 1
            if (
                self.project_table.current_table == self.project_table.CurrentTable.DATASETS
                and self.splits_widget.train_val_splits.is_hidden()
            ):
                self.select_all_datasets_checkbox.hide()
                back_btn.hide()
                self.project_table.switch_table(self.project_table.CurrentTable.PROJECTS)
                self.project_table.table.clear_selection()
                self.project_table.enable()
                next_btn.text = "Next"
                next_btn.disable()
            # Step 3 -> Step 2
            elif (
                self.project_table.current_table == self.project_table.CurrentTable.DATASETS
                and not self.splits_widget.train_val_splits.is_hidden()
            ):
                next_btn.text = "Next"
                next_btn.enable()
                self.splits_widget.train_val_splits.hide()
                self.project_table.switch_table(self.project_table.CurrentTable.DATASETS)
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
        self.project_table.hide()
        self.splits_widget.train_val_splits.show()
        project_id = self.get_selected_project_id()
        if not project_id:
            raise RuntimeError("Project ID is not selected. Cannot set splits data.")
        dataset_ids = self.get_selected_dataset_ids()
        try:
            self.splits_widget.train_val_splits._content.loading = True
            self.splits_widget.set_project_id(project_id, dataset_ids)
            self.splits_widget._detect_splits(True, True, True)
        finally:
            self.splits_widget.train_val_splits._content.loading = False

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
        dataset_ids = self.project_table.get_selected_dataset_ids()
        self.splits_widget.train_val_splits._dataset_ids = dataset_ids

        settings_data = {
            "workspace_id": self.project_table.team_workspace_selector.get_selected_workspace_id(),
            "team_id": self.project_table.team_workspace_selector.get_selected_team_id(),
            "project_id": self.get_selected_project_id(),
        }

        for callback in self._on_settings_saved_callbacks:
            try:
                callback(settings_data)
            except Exception as e:
                logger.error(f"Error in settings saved callback: {e}")

    def _get_splits_downloadless(self):
        import random

        from supervisely.annotation.annotation import Annotation
        from supervisely.api.entities_collection_api import CollectionTypeFilter
        from supervisely.project.project_meta import ProjectMeta

        if not self._project_id:
            return [], []

        dataset_ids = None
        if hasattr(self, "_dataset_ids"):
            dataset_ids = self._dataset_ids

        if dataset_ids is None:
            raise RuntimeError("Cannot get splits: dataset_ids is None")

        filters = [{"field": "id", "operator": "in", "value": dataset_ids}]

        split_method = self._content.get_active_tab()
        train_set, val_set = [], []
        if split_method == "Random":
            splits_counts = self._random_splits_table.get_splits_counts()
            train_count = splits_counts["train"]
            val_count = splits_counts["val"]
            val_part = val_count / (val_count + train_count)

            dataset_infos = self._api.dataset.get_list(
                self._project_id, filters=filters, recursive=True
            )

            all_images = []
            for ds_info in dataset_infos:
                img_infos = self._api.image.get_list(ds_info.id)
                all_images.extend(img_infos)

            random.shuffle(all_images)
            split_idx = round(len(all_images) * (1 - val_part))
            train_set = all_images[:split_idx]
            val_set = all_images[split_idx:]
        elif split_method == "Based on item tags":
            train_tag_name = self._train_tag_select.get_selected_name()
            val_tag_name = self._val_tag_select.get_selected_name()
            add_untagged_to = self._untagged_select.get_value()
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(self._project_id))
            for ds_info in self._api.dataset.get_list(self._project_id, recursive=True):
                ann_infos = self._api.annotation.get_list(ds_info.id)
                anns = [Annotation.from_json(ann.annotation, project_meta) for ann in ann_infos]
                for ann_info, ann in zip(ann_infos, anns):
                    tags = [tag.name for tag in ann.img_tags]
                    if train_tag_name in tags:
                        train_set.append(self._api.image.get_info_by_id(ann_info.image_id))
                    elif val_tag_name in tags:
                        val_set.append(self._api.image.get_info_by_id(ann_info.image_id))
                    elif add_untagged_to == "train":
                        train_set.append(self._api.image.get_info_by_id(ann_info.image_id))
                    elif add_untagged_to == "val":
                        val_set.append(self._api.image.get_info_by_id(ann_info.image_id))
        elif split_method == "Based on datasets":
            train_ds_ids = self._train_ds_select.get_selected_ids()
            val_ds_ids = self._val_ds_select.get_selected_ids()
            ds_infos = self._api.dataset.get_list(self._project_id, filters=filters, recursive=True)
            for ds_info in ds_infos:
                if ds_info.id in train_ds_ids:
                    train_set.extend(self._api.image.get_list(ds_info.id))
                if ds_info.id in val_ds_ids:
                    val_set.extend(self._api.image.get_list(ds_info.id))
        elif split_method == "Based on collections":
            train_collections = self._train_collections_select.get_selected_ids()
            val_collections = self._val_collections_select.get_selected_ids()

            for collection_ids, items_list in [
                (train_collections, train_set),
                (val_collections, val_set),
            ]:
                for collection_id in collection_ids:
                    collection_items = self._api.entities_collection.get_items(
                        collection_id=collection_id,
                        project_id=self._project_id,
                        collection_type=CollectionTypeFilter.DEFAULT,
                    )
                    items_list.extend(collection_items)
        return train_set, val_set
