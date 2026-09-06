# pylint: skip-file

import os
import shutil
import sys
import unittest

import numpy as np

from supervisely.api.api import Api
from supervisely.app.widgets.train_val_splits.train_val_splits import TrainValSplits

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)


class TestTrainValSplitsNestedDatasets(unittest.TestCase):
    """
    Regression test for a bug where "Based on datasets" split raised
    KeyError: "Dataset '<name>' not found" whenever the selected dataset had a
    parent (i.e. was nested). TrainValSplits.get_splits() built dataset names
    from api.dataset.tree() using only the short dataset name and dropped the
    parents path, while the locally downloaded Project keys nested datasets by
    their full name (parents + short name). See report from customer running
    Train YOLO app v1.0.46 on datasets structured as:
        Project / TrainData / Dataset1
        Project / TrainData / Dataset2 / subfolder1
        Project / TrainData / Dataset2 / subfolder2
        Project / ValData / Dataset3
        Project / ValData / Dataset4
    """

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        cls.workspace_id = 1
        # SelectDatasetTree (used by TrainValSplits for the "Based on datasets" tab)
        # reads the team id from the environment, which is normally injected by the
        # platform when an app is running inside it.
        os.environ["TEAM_ID"] = str(cls.api.team.get_list()[0].id)
        os.environ["WORKSPACE_ID"] = str(cls.workspace_id)
        app_dir = os.path.join(sdk_path, "tests", "api_tests", "_tmp_app_data")
        os.makedirs(app_dir, exist_ok=True)
        os.environ["DEBUG_APP_DIR"] = app_dir

        cls.project_info = cls.api.project.create(
            cls.workspace_id, "Test nested train val split", change_name_if_conflict=True
        )

        cls.train_data = cls.api.dataset.create(cls.project_info.id, "TrainData")
        cls.val_data = cls.api.dataset.create(cls.project_info.id, "ValData")
        cls.dataset1 = cls.api.dataset.create(
            cls.project_info.id, "Dataset1", parent_id=cls.train_data.id
        )
        cls.dataset2 = cls.api.dataset.create(
            cls.project_info.id, "Dataset2", parent_id=cls.train_data.id
        )
        cls.subfolder1 = cls.api.dataset.create(
            cls.project_info.id, "subfolder1", parent_id=cls.dataset2.id
        )
        cls.subfolder2 = cls.api.dataset.create(
            cls.project_info.id, "subfolder2", parent_id=cls.dataset2.id
        )
        cls.dataset3 = cls.api.dataset.create(
            cls.project_info.id, "Dataset3", parent_id=cls.val_data.id
        )
        cls.dataset4 = cls.api.dataset.create(
            cls.project_info.id, "Dataset4", parent_id=cls.val_data.id
        )
        # a flat (non-nested) dataset, to make sure the fix does not break the
        # already-working case of top-level datasets
        cls.flat_dataset = cls.api.dataset.create(cls.project_info.id, "FlatDataset")

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        cls.leaf_image_counts = {
            cls.dataset1.id: 3,
            cls.subfolder1.id: 2,
            cls.subfolder2.id: 4,
            cls.dataset3.id: 5,
            cls.dataset4.id: 1,
            cls.flat_dataset.id: 6,
        }
        for ds_id, count in cls.leaf_image_counts.items():
            names = [f"img_{ds_id}_{i}.png" for i in range(count)]
            imgs = [img] * count
            cls.api.image.upload_nps(ds_id, names, imgs)

    @classmethod
    def tearDownClass(cls):
        cls.api.project.remove(cls.project_info.id)
        shutil.rmtree(os.environ["DEBUG_APP_DIR"], ignore_errors=True)

    def _make_widget(self) -> TrainValSplits:
        widget = TrainValSplits(
            project_id=self.project_info.id,
            random_splits=False,
            tags_splits=False,
            datasets_splits=True,
        )
        # In the real app the dataset tree is populated once the user picks a
        # project in its (independent) project selector. TrainValSplits never
        # calls this itself, so it has to be done explicitly here to be able to
        # select datasets programmatically, same as the widget's own
        # set_dataset_ids()/get_item_by_id() require populated tree items.
        widget._train_ds_select.set_project_id(self.project_info.id)
        widget._val_ds_select.set_project_id(self.project_info.id)
        return widget

    def test_split_by_nested_datasets_does_not_raise_keyerror(self):
        train_ids = [self.dataset1.id, self.subfolder1.id, self.subfolder2.id]
        val_ids = [self.dataset3.id, self.dataset4.id]

        widget = self._make_widget()
        widget.set_datasets_splits(train_ids, val_ids)

        train_set, val_set = widget.get_splits()

        expected_train = (
            self.leaf_image_counts[self.dataset1.id]
            + self.leaf_image_counts[self.subfolder1.id]
            + self.leaf_image_counts[self.subfolder2.id]
        )
        expected_val = self.leaf_image_counts[self.dataset3.id] + self.leaf_image_counts[self.dataset4.id]

        self.assertEqual(len(train_set), expected_train)
        self.assertEqual(len(val_set), expected_val)

    def test_split_by_mixed_flat_and_nested_datasets(self):
        # flat top-level dataset + a deeply nested one selected together
        train_ids = [self.flat_dataset.id, self.subfolder2.id]
        val_ids = [self.dataset4.id]

        widget = self._make_widget()
        widget.set_datasets_splits(train_ids, val_ids)

        train_set, val_set = widget.get_splits()

        expected_train = (
            self.leaf_image_counts[self.flat_dataset.id] + self.leaf_image_counts[self.subfolder2.id]
        )
        expected_val = self.leaf_image_counts[self.dataset4.id]

        self.assertEqual(len(train_set), expected_train)
        self.assertEqual(len(val_set), expected_val)


if __name__ == "__main__":
    unittest.main()
