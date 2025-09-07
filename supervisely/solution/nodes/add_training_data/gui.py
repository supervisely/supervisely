from supervisely.app.content import DataJson
from supervisely.app.widgets import (
    Button,
    Container,
    Dialog,
    Text,
    Widget,
    ProjectDatasetTable,
    Flexbox
)
from supervisely.nn.training.gui.train_val_splits_selector import TrainValSplitsSelector
from supervisely.project.project import ProjectType
from typing import Callable, List, Any
from supervisely import logger
from supervisely.api.api import Api
from supervisely.project import ProjectMeta

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
    def splits_selector(self) -> TrainValSplitsSelector:
        if not hasattr(self, "_splits_selector"):
            self._splits_selector = TrainValSplitsSelector(self.api, self.get_selected_project_id())
            self.splits_selector.container.hide()
        return self._splits_selector

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

        tables_container = Container([self.project_table, self.splits_selector.container], gap=0)
        btns_container = self._init_btns()

        return Container([desc, tables_container, btns_container], gap=20)

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
            if self.project_table.is_hidden():
                self._trigger_settings_saved_event()
                self.modal.hide()
                return
            
            if self.project_table.current_table == self.project_table.CurrentTable.PROJECTS:
                self.project_table.switch_table(self.project_table.CurrentTable.DATASETS)
                back_btn.show()
            elif self.project_table.current_table == self.project_table.CurrentTable.DATASETS:
                self.splits_selector.set_project_id(self.get_selected_project_id())
                self._set_split_selector_active_tab()
                self.splits_selector.container.show()
                self.project_table.hide()
                next_btn.text = "Select"

        @back_btn.click
        def on_back_btn_click():
            if not self.splits_selector.container.is_hidden():
                self.splits_selector.container.hide()
                self.project_table.show()
                next_btn.text = "Next"

            if self.project_table.current_table == self.project_table.CurrentTable.DATASETS:
                self.project_table.switch_table(self.project_table.CurrentTable.PROJECTS)
                back_btn.hide()

        return Flexbox([Container([back_btn, next_btn], direction="horizontal")], horizontal_alignment="flex-end")

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
        splits = self.splits_selector.train_val_splits.get_splits()
        settings_data = {
            'project_id': self.get_selected_project_id(),
            'dataset_ids': self.get_selected_dataset_ids(),
            'splits': splits,
        }
        
        for callback in self._on_settings_saved_callbacks:
            try:
                callback(settings_data)
            except Exception as e:
                logger.error(f"Error in settings saved callback: {e}")

    def _set_split_selector_active_tab(self):
        # First, check if we can set tab to collections: if there are collections with train/val names in the project, set tab to collections
        # If not, check if there are datasets with train/val names in the project, set tab to datasets
        # If neither, check if there are item tags with train/val names in the project, set tab to item tags
        # Otherwise, default to random split tab

        project_id = self.get_selected_project_id()
        collections = self.api.entities_collection.get_list(project_id)
        collection_names = [col.name.lower() for col in collections]
        if "train" in collection_names and "val" in collection_names:
            self.splits_selector.train_val_splits.set_split_method("collections")
            return
        
        datasets = self.api.dataset.get_list(project_id)
        dataset_names = [ds.name.lower() for ds in datasets]
        if "train" in dataset_names and "val" in dataset_names:
            self.splits_selector.train_val_splits.set_split_method("datasets")
            return
        
        meta = ProjectMeta.from_json(self.api.project.get_meta(project_id))
        tag_metas = meta.tag_metas.items()
        item_tag_names = [tag.name.lower() for tag in tag_metas]
        if "train" in item_tag_names and "val" in item_tag_names:
            self.splits_selector.train_val_splits.set_split_method("tags")
            return

        self.splits_selector.train_val_splits.set_split_method("random")