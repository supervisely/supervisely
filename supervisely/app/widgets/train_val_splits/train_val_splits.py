import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import supervisely as sly
from supervisely._utils import rand_str
from supervisely.api.api import Api
from supervisely.app import DataJson, StateJson, get_data_dir
from supervisely.app.widgets import (
    Container,
    Field,
    NotificationBox,
    RadioTabs,
    SelectString,
    SelectTagMeta,
    Widget,
)
from supervisely.app.widgets.random_splits_table.random_splits_table import (
    RandomSplitsTable,
)
from supervisely.app.widgets.select_dataset_tree.select_dataset_tree import (
    SelectDatasetTree,
)
from supervisely.io.fs import remove_dir
from supervisely.project import get_project_class
from supervisely.project.pointcloud_episode_project import PointcloudEpisodeProject
from supervisely.project.pointcloud_project import PointcloudProject
from supervisely.project.project import ItemInfo, Project
from supervisely.project.project_type import ProjectType
from supervisely.project.video_project import VideoProject
from supervisely.project.volume_project import VolumeProject


class TrainValSplits(Widget):
    def __init__(
        self,
        project_id: Optional[int] = None,
        project_fs: Optional[
            Union[Project, VideoProject, VolumeProject, PointcloudProject, PointcloudEpisodeProject]
        ] = None,
        random_splits: Optional[bool] = True,
        tags_splits: Optional[bool] = True,
        datasets_splits: Optional[bool] = True,
        widget_id: Optional[int] = None,
    ):
        self._project_id = project_id
        self._project_fs = project_fs

        if project_fs is not None and project_id is not None:
            raise ValueError(
                "You can not provide both project_id and project_fs parameters to TrainValSplits widget."
            )
        if project_fs is None and project_id is None:
            raise ValueError(
                "You should provide at least one of: project_id or project_fs parameters to TrainValSplits widget."
            )

        self._project_info = None
        if project_id is not None:
            self._api = Api()
            self._project_info = self._api.project.get_info_by_id(
                self._project_id, raise_error=True
            )

        self._project_type = project_fs.type if project_id is None else self._project_info.type
        self._project_class = get_project_class(self._project_type)

        self._random_splits_table: RandomSplitsTable = None
        self._train_tag_select: SelectTagMeta = None
        self._val_tag_select: SelectTagMeta = None
        self._untagged_select: SelectString = None
        self._train_ds_select: Union[SelectDatasetTree, SelectString] = None
        self._val_ds_select: Union[SelectDatasetTree, SelectString] = None
        self._split_methods = []

        contents = []
        tabs_descriptions = []
        if random_splits:
            self._split_methods.append("Random")
            tabs_descriptions.append("Shuffle data and split with defined probability")
            contents.append(self._get_random_content())
        if tags_splits:
            self._split_methods.append("Based on item tags")
            tabs_descriptions.append(f"{self._project_type.capitalize()} should have assigned train or val tag")
            contents.append(self._get_tags_content())
        if datasets_splits:
            self._split_methods.append("Based on datasets")
            tabs_descriptions.append("Select one or several datasets for every split")
            contents.append(self._get_datasets_content())
        if not self._split_methods:
            raise ValueError(
                "Any of split methods [random_splits, tags_splits, datasets_splits] must be specified in TrainValSplits."
            )

        self._content = RadioTabs(
            titles=self._split_methods,
            descriptions=tabs_descriptions,
            contents=contents,
        )

        super().__init__(widget_id=widget_id, file_path=__file__)

    def _get_random_content(self):
        items_count = 0
        if self._project_id is not None:
            items_count = self._project_info.items_count
        elif self._project_fs is not None:
            items_count = self._project_fs.total_items
        self._random_splits_table = RandomSplitsTable(items_count)

        return Container(widgets=[self._random_splits_table], direction="vertical", gap=5)

    def _get_tags_content(self):
        notification_box = NotificationBox(
            title="Notice: How to make equal splits",
            description="Choose the same tag for train/validation to make splits equal. Can be used for debug and for tiny projects",
            box_type="info",
        )
        if self._project_id is not None:
            self._train_tag_select = SelectTagMeta(project_id=self._project_id, show_label=False)
            self._val_tag_select = SelectTagMeta(project_id=self._project_id, show_label=False)
        elif self._project_fs is not None:
            self._train_tag_select = SelectTagMeta(
                project_meta=self._project_fs.meta, show_label=False
            )
            self._val_tag_select = SelectTagMeta(
                project_meta=self._project_fs.meta, show_label=False
            )
        self._untagged_select = SelectString(
            values=["train", "val", "ignore"],
            labels=[
                f"add untagged {self._project_type} to train set",
                f"add untagged {self._project_type} to val set",
                f"ignore untagged {self._project_type}",
            ],
            placeholder="Select action",
        )
        train_field = Field(
            self._train_tag_select,
            title="Train tag",
            description=f"all {self._project_type} with this tag are considered as training set",
        )
        val_field = Field(
            self._val_tag_select,
            title="Validation tag",
            description=f"all {self._project_type} with this tag are considered as validation set",
        )
        without_tags_field = Field(
            self._untagged_select,
            title=f"{self._project_type.capitalize()} without selected tags",
            description=f"Choose what to do with untagged {self._project_type}",
        )
        return Container(
            widgets=[
                notification_box,
                train_field,
                val_field,
                without_tags_field,
            ],
            direction="vertical",
            gap=5,
        )

    def _get_datasets_content(self):
        notification_box = NotificationBox(
            title="Notice: How to make equal splits",
            description="Choose the same dataset(s) for train/validation to make splits equal. Can be used for debug and for tiny projects",
            box_type="info",
        )
        if self._project_id is not None:
            self._train_ds_select = SelectDatasetTree(
                multiselect=True,
                flat=True,
                select_all_datasets=False,
                allowed_project_types=[self._project_type],
                always_open=False,
                compact=True,
                team_is_selectable=False,
                workspace_is_selectable=False,
                append_to_body=True,
            )

            self._val_ds_select = SelectDatasetTree(
                multiselect=True,
                flat=True,
                select_all_datasets=False,
                allowed_project_types=[self._project_type],
                always_open=False,
                compact=True,
                team_is_selectable=False,
                workspace_is_selectable=False,
                append_to_body=True,
            )

            # old implementation
            # self._train_ds_select = SelectDataset(
            #     project_id=self._project_id, multiselect=True, compact=True, show_label=False
            # )
            # self._val_ds_select = SelectDataset(
            #     project_id=self._project_id, multiselect=True, compact=True, show_label=False
            # )
        elif self._project_fs is not None:
            ds_names = [ds.name for ds in self._project_fs.datasets]
            self._train_ds_select = SelectString(ds_names, multiple=True)
            self._val_ds_select = SelectString(ds_names, multiple=True)
        train_field = Field(
            self._train_ds_select,
            title="Train dataset(s)",
            description=f"all {self._project_type} in selected dataset(s) are considered as training set",
        )
        val_field = Field(
            self._val_ds_select,
            title="Validation dataset(s)",
            description=f"all {self._project_type} in selected dataset(s) are considered as validation set",
        )
        return Container(
            widgets=[notification_box, train_field, val_field], direction="vertical", gap=5
        )

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}

    def get_splits(self) -> Tuple[List[ItemInfo], List[ItemInfo]]:
        split_method = self._content.get_active_tab()
        tmp_project_dir = None
        train_set, val_set = [], []
        if self._project_fs is None:
            tmp_project_dir = os.path.join(get_data_dir(), rand_str(15))
            self._project_class.download(self._api, self._project_id, tmp_project_dir)

        project_dir = tmp_project_dir if tmp_project_dir is not None else self._project_fs.directory

        if split_method == "Random":
            splits_counts = self._random_splits_table.get_splits_counts()
            train_count = splits_counts["train"]
            val_count = splits_counts["val"]
            val_part = val_count / (val_count + train_count)
            project = self._project_class(project_dir, sly.OpenMode.READ)
            n_images = project.total_items
            new_val_count = round(val_part * n_images)
            new_train_count = n_images - new_val_count

            train_set, val_set = self._project_class.get_train_val_splits_by_count(
                project_dir, new_train_count, new_val_count
            )

        elif split_method == "Based on item tags":
            train_tag_name = self._train_tag_select.get_selected_name()
            val_tag_name = self._val_tag_select.get_selected_name()
            add_untagged_to = self._untagged_select.get_value()
            train_set, val_set = self._project_class.get_train_val_splits_by_tag(
                project_dir, train_tag_name, val_tag_name, add_untagged_to
            )

        elif split_method == "Based on datasets":
            if self._project_id is not None:
                self._train_ds_select: SelectDatasetTree
                self._val_ds_select: SelectDatasetTree
                train_ds_ids = self._train_ds_select.get_selected_ids()
                val_ds_ids = self._val_ds_select.get_selected_ids()
                ds_infos = [dataset for _, dataset in self._api.dataset.tree(self._project_id)]
                train_ds_names, val_ds_names = [], []
                for ds_info in ds_infos:
                    if ds_info.id in train_ds_ids:
                        train_ds_names.append(ds_info.name)
                    if ds_info.id in val_ds_ids:
                        val_ds_names.append(ds_info.name)
            elif self._project_fs is not None:
                self._train_ds_select: SelectString
                self._val_ds_select: SelectString
                train_ds_names = self._train_ds_select.get_value()
                val_ds_names = self._val_ds_select.get_value()

            train_set, val_set = self._project_class.get_train_val_splits_by_dataset(
                project_dir, train_ds_names, val_ds_names
            )

        if tmp_project_dir is not None:
            remove_dir(tmp_project_dir)
        return train_set, val_set

    def set_split_method(self, split_method: Literal["random", "tags", "datasets"]):
        if split_method == "random":
            split_method = "Random"
        elif split_method == "tags":
            split_method = "Based on item tags"
        elif split_method == "datasets":
            split_method = "Based on datasets"
        self._content.set_active_tab(split_method)
        StateJson().send_changes()
        DataJson().send_changes()

    def get_split_method(self) -> str:
        return self._content.get_active_tab()

    def set_random_splits(
        self, split: Literal["train", "training", "val", "validation"], percent: int
    ):
        self._content.set_active_tab("Random")
        if split == "train" or split == "training":
            self._random_splits_table.set_train_split_percent(percent)
        elif split == "val" or split == "validation":
            self._random_splits_table.set_val_split_percent(percent)
        else:
            raise ValueError("Split value must be 'train', 'training', 'val' or 'validation'")

    def get_train_split_percent(self) -> int:
        return self._random_splits_table.get_train_split_percent()

    def get_val_split_percent(self) -> int:
        return 100 - self._random_splits_table.get_train_split_percent()

    def set_tags_splits(
        self, train_tag: str, val_tag: str, untagged_action: Literal["train", "val", "ignore"]
    ):
        self._content.set_active_tab("Based on item tags")
        self._train_tag_select.set_name(train_tag)
        self._val_tag_select.set_name(val_tag)
        self._untagged_select.set_value(untagged_action)

    def get_train_tag(self) -> str:
        return self._train_tag_select.get_selected_name()

    def get_val_tag(self) -> str:
        return self._val_tag_select.get_selected_name()

    def set_datasets_splits(self, train_datasets: List[int], val_datasets: List[int]):
        self._content.set_active_tab("Based on datasets")
        self._train_ds_select.set_dataset_ids(train_datasets)
        self._val_ds_select.set_dataset_ids(val_datasets)

    def get_train_dataset_ids(self) -> List[int]:
        return self._train_ds_select.get_selected_ids()

    def get_val_dataset_ids(self) -> List[int]:
        return self._val_ds_select.get_selected_ids()

    def get_untagged_action(self) -> str:
        return self._untagged_select.get_value()

    def disable(self):
        self._content.disable()
        if self._random_splits_table is not None:
            self._random_splits_table.disable()
        if self._train_tag_select is not None:
            self._train_tag_select.disable()
        if self._val_tag_select is not None:
            self._val_tag_select.disable()
        if self._untagged_select is not None:
            self._untagged_select.disable()
        if self._train_ds_select is not None:
            self._train_ds_select.disable()
        if self._val_ds_select is not None:
            self._val_ds_select.disable()
        self._disabled = True
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        self._content.enable()
        if self._random_splits_table is not None:
            self._random_splits_table.enable()
        if self._train_tag_select is not None:
            self._train_tag_select.enable()
        if self._val_tag_select is not None:
            self._val_tag_select.enable()
        if self._untagged_select is not None:
            self._untagged_select.enable()
        if self._train_ds_select is not None:
            self._train_ds_select.enable()
        if self._val_ds_select is not None:
            self._val_ds_select.enable()
        self._disabled = False
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()
