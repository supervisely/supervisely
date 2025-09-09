from typing import Any, Callable, List

from supervisely import logger
from supervisely.api.api import Api
from supervisely.app.widgets import (
    Button,
    CheckboxField,
    Container,
    Dialog,
    Flexbox,
    ProjectDatasetTable,
    Text,
    Widget,
)
from supervisely.project.project import ProjectType


# @TODO: move logic from Data Commander because of bugs.
# Currently, it is impossible to copy several nested datasets if they do not share a parent, which leads to unexpected behavior.
# Also, at the moment it is very tricky to get resulting DatasetInfos as they are being created externally - Not implemented yet.
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
            self._project_table = ProjectDatasetTable(allowed_project_types=[ProjectType.IMAGES])
        return self._project_table

    @property
    def replicate_structure_checkbox(self) -> CheckboxField:
        if not hasattr(self, "_replicate_structure_checkbox"):
            self._replicate_structure_checkbox = CheckboxField(
                title="Replicate Structure",
                description="Replicate the dataset structure of the source project.",
            )
            self._replicate_structure_checkbox.hide()
        return self._replicate_structure_checkbox

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

        return Container(
            [desc, self.project_table, self.replicate_structure_checkbox, btns_container], gap=20
        )

    def _init_btns(self) -> Container:
        next_btn = Button(
            "Next",
            button_size="small",
            icon="zmdi zmdi-arrow-right",
            style="primary",
        )
        back_btn = Button("Back", plain=True, icon="zmdi zmdi-arrow-left", button_size="small")
        back_btn.hide()

        @self.project_table.table.selection_changed
        def on_table_selection_change(selected_items):
            if selected_items:
                next_btn.enable()
            else:
                next_btn.disable()

        @next_btn.click
        def on_next_btn_click():
            if self.project_table.current_table == self.project_table.CurrentTable.PROJECTS:
                self.replicate_structure_checkbox.show()
                self.project_table.switch_table(self.project_table.CurrentTable.DATASETS)
                back_btn.show()
                next_btn.text = "Add"
            elif self.project_table.current_table == self.project_table.CurrentTable.DATASETS:
                self._trigger_settings_saved_event()
                self.modal.hide()
                next_btn.text = "Next"
                self.project_table.table.clear_selection()
                self.replicate_structure_checkbox.hide()
                self.project_table.switch_table(self.project_table.CurrentTable.PROJECTS)

        @back_btn.click
        def on_back_btn_click():
            if self.project_table.current_table == self.project_table.CurrentTable.DATASETS:
                self.replicate_structure_checkbox.hide()
                back_btn.hide()
                self.project_table.switch_table(self.project_table.CurrentTable.PROJECTS)

        return Flexbox(
            [Container([back_btn, next_btn], direction="horizontal")],
            horizontal_alignment="flex-end",
        )

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
        rows = sorted(self.project_table.table.get_selected_rows(), key=lambda x: x.row[1])
        for ds_info, row in zip(ds_infos, rows):
            full_name: str = row[0]
            last_name = ds_info.name
            name = full_name.removesuffix(last_name).rstrip("/")
            selected_ids_to_parents[ds_info.id] = name.split("/")

        settings_data = {
            "workspace_id": self.project_table.team_workspace_selector.get_selected_workspace_id(),
            "team_id": self.project_table.team_workspace_selector.get_selected_team_id(),
            "project_id": self.get_selected_project_id(),
            "dataset_ids": self.get_selected_dataset_ids(),
            "replicate_structure": self.replicate_structure_checkbox.is_checked(),
            "selected_ids_to_parents": selected_ids_to_parents,
        }

        for callback in self._on_settings_saved_callbacks:
            try:
                callback(settings_data)
            except Exception as e:
                logger.error(f"Error in settings saved callback: {e}")
