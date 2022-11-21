from typing import List, Optional, Dict
from supervisely import Project, Api
from supervisely.app import StateJson
from supervisely.app.widgets import Widget, RadioTabs, Container, NotificationBox, SelectDataset, SelectString, Field, SelectTagMeta
from supervisely.app.widgets.random_splits_table.random_splits_table import RandomSplitsTable

class TrainValSplits(Widget):
    def __init__(
        self,
        project_id: Optional[int] = None,
        project_fs: Optional[Project] = None,
        random_splits: Optional[bool] = True,
        tags_splits: Optional[bool] = True,
        datasets_splits: Optional[bool] = True,
        widget_id: Optional[int] = None,
    ):
        self._project_id = project_id
        self._project_fs: Project = project_fs
        if project_fs is not None and project_id is not None:
            raise ValueError("You can not provide both project_id and project_fs parameters to TrainValSplits widget.")
        if project_fs is None and project_id is None:
            raise ValueError("You should provide at least one of: project_id or project_fs parameters to TrainValSplits widget.")
        
        if project_id is not None:
            self._api = Api()
        self._split_methods = []
        contents = []
        if random_splits:
            self._split_methods.append('Random')
            contents.append(self._get_random_content())
        if tags_splits:
            self._split_methods.append('Based on item tags')
            contents.append(self._get_tags_content())
        if datasets_splits:
            self._split_methods.append('Based on datasets')
            contents.append(self._get_datasets_content())
        if not self._split_methods:
            raise ValueError("Any of split methods [random_splits, tags_splits, datasets_splits] must be specified in TrainValSplits.")
        
        self._content = RadioTabs(
            titles=self._split_methods,
            contents=contents,
        )

        super().__init__(widget_id=widget_id, file_path=__file__)
    
    def _get_random_content(self):
        if self._project_id is not None:
            info = self._api.project.get_info_by_id(self._project_id)
            items_count = info.items_count
        elif self._project_fs is not None:
            items_count = self._project_fs.total_items
        splits_table = RandomSplitsTable(items_count)
        
        return Container(
            widgets=[splits_table],
            direction='vertical',
            gap = 5
        )
    
    def _get_tags_content(self):
        notification_box = NotificationBox(
            title="Notice: How to make equal splits",
            description="Choose the same tag for train/validation to make splits equal. Can be used for debug and for tiny projects",
            box_type='info'
        )
        if self._project_id is not None:
            train_select = SelectTagMeta(project_id=self._project_id, show_label=False)
            val_select = SelectTagMeta(project_id=self._project_id, show_label=False)
        elif self._project_fs is not None:
            train_select = SelectTagMeta(project_meta=self._project_fs.meta, show_label=False)
            val_select = SelectTagMeta(project_meta=self._project_fs.meta, show_label=False)
        without_tags_select = SelectString(
            values = ["train", "val", "ignore"],
            labels=[
                "add untagged images to train set",
                "add untagged images to val set",
                "ignore untagged images", 
            ],
            placeholder="Select action"
        )
        train_field = Field(
            train_select, 
            title="Train tag",
            description="all images with this tag are considered as training set"
        )
        val_field = Field(
            val_select,
            title="Validation tag",
            description="all images with this tag are considered as validation set"
        )
        without_tags_field = Field(
            without_tags_select,
            title="Images without selected tags",
            description="Choose what to do with untagged images"
        )
        return Container(
            widgets=[
                notification_box,
                train_field,
                val_field,
                without_tags_field,
            ],
            direction='vertical',
            gap = 5
        )

    
    def _get_datasets_content(self):
        notification_box = NotificationBox(
            title="Notice: How to make equal splits",
            description="Choose the same dataset(s) for train/validation to make splits equal. Can be used for debug and for tiny projects",
            box_type='info'
        )
        if self._project_id is not None:
            train_select = SelectDataset(
                project_id=self._project_id, 
                multiselect=True, 
                compact=True, 
                show_label=False
            )
            val_select = SelectDataset(
                project_id=self._project_id, 
                multiselect=True, 
                compact=True, 
                show_label=False
            )
        elif self._project_fs is not None:
            train_select = SelectDataset(
                project_meta=self._project_fs.meta, 
                multiselect=True, 
                compact=True, 
                show_label=False
            )
            val_select = SelectDataset(
                project_meta=self._project_fs.meta,
                multiselect=True, 
                compact=True, 
                show_label=False
            )
        train_field = Field(
            train_select, 
            title="Train dataset(s)",
            description="all images in selected dataset(s) are considered as training set"
        )
        val_field = Field(
            val_select,
            title="Validation dataset(s)",
            description="all images in selected dataset(s) are considered as validation set"
        )
        return Container(
            widgets=[notification_box, train_field, val_field],
            direction='vertical',
            gap = 5)

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}