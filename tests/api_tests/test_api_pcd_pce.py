# isort:skip_file
import os
import sys
import unittest

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)
from unittest.mock import patch

import supervisely as sly
from tqdm import tqdm

from supervisely.api.api import Api
from supervisely.api.pointcloud.pointcloud_api import PointcloudInfo
from supervisely.io.fs import get_file_name_with_ext, list_files, list_files_recursively


class TestPCDApi(unittest.TestCase):
    project_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        # workspace_id = input("Enter workspace ID for tests here >> ")
        workspace_id = 1051
        with patch("builtins.input", return_value="user_input_value"):
            project = cls.api.project.create(
                workspace_id=int(workspace_id),
                name="[UT] PCD api",
                type=sly.ProjectType.POINT_CLOUDS,
                change_name_if_conflict=True,
            )
        cls.project_id = project.id

    @classmethod
    def tearDownClass(cls):
        cls.api.project.remove_permanently(cls.project_id)

    def setUp(self):
        # Set up any necessary test data or configurations
        self.pointcloud_api = self.api.pointcloud
        self.dataset_instance = self.api.dataset
        self.files_path_1 = "/home/ganpoweird/Work/test_assets/pcds/pointcloud"
        self.files_path_2 = "/test_assets/pcds_2"
        self.files_path_3 = "/test_assets/pcds_3"
        self.all_paths = [self.files_path_1, self.files_path_2, self.files_path_3]

    def create_test_datasets(self, count):
        """
        Create test datasets.
        """
        created_dataset_ids = []
        for i in range(count):
            created_dataset = self.dataset_instance.create(
                self.project_id,
                name=f"[UT] Dataset pcd {i+1}",
                change_name_if_conflict=True,
            )
            created_dataset_ids.append(created_dataset.id)
        return created_dataset_ids

    def test_upload_dirs(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        pcd_paths = list_files_recursively(self.files_path_1)
        names = [get_file_name_with_ext(pcd_path) for pcd_path in pcd_paths]
        # progress_cb = tqdm(total=len(pcd_paths), desc="Uploading pcds")
        progress_cb = tqdm(total=100, desc="Uploading pcds")
        # Call the method being tested
        pcd_info = self.pointcloud_api.upload_dirs(
            dataset_id, [self.files_path_1], progress_cb=progress_cb
        )

        # Verify the method returns the correct value
        self.assertIsInstance(pcd_info, list)
        self.assertEqual(len(pcd_info), len(pcd_paths))
        for info in pcd_info:
            self.assertIsInstance(info, PointcloudInfo)
            self.assertIsInstance(info.id, int)
            self.assertEqual(info.dataset_id, dataset_id)

        with self.assertRaises(ValueError):
            self.pointcloud_api.upload_dir(
                dataset_id, self.files_path_1, change_name_if_conflict=False
            )

        pcd_info = self.pointcloud_api.upload_dirs(
            dataset_id, [self.files_path_1], progress_cb=progress_cb
        )


if __name__ == "__main__":
    unittest.main()
