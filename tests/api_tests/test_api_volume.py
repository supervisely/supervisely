import os
import sys
import unittest

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)
from unittest.mock import MagicMock, patch

from supervisely.api.api import Api
from supervisely.api.volume.volume_api import VolumeInfo
from supervisely.io.fs import get_file_name_with_ext, list_files, list_files_recursively

import supervisely as sly  # isort:skip
from tqdm import tqdm  # isort:skip


class TestVolumeApi(unittest.TestCase):
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
                name="[UT] Volume api",
                type=sly.ProjectType.VOLUMES,
                change_name_if_conflict=True,
            )
        cls.project_id = project.id

    @classmethod
    def tearDownClass(cls):
        cls.api.project.remove_permanently(cls.project_id)

    def setUp(self):
        self.volume_api = self.api.volume
        self.dataset_instance = self.api.dataset
        self.files_path_1 = "/home/ganpoweird/Work/test_assets/volumes_1"
        self.files_path_2 = "/test_assets/volumes_2"
        self.files_path_3 = "/test_assets/volumes_3"
        self.all_paths = [self.files_path_1, self.files_path_2, self.files_path_3]

    def create_test_datasets(self, count):
        """
        Create test datasets.
        """
        created_dataset_ids = []
        for i in range(count):
            created_dataset = self.dataset_instance.create(
                self.project_id,
                name=f"[UT] Dataset volume {i+1}",
                change_name_if_conflict=True,
            )
            created_dataset_ids.append(created_dataset.id)
        return created_dataset_ids

    def test_upload_dirs_and_dir(self):
        dataset_id = self.create_test_datasets(1)[0]
        volumes = 6  #! determined by the number of volumes in the test directory

        progress_cb = tqdm(total=volumes, desc="Uploading volumes")

        # Call the method being tested
        volumes_info = self.volume_api.upload_dirs(
            dataset_id, [self.files_path_1], progress_cb=progress_cb, log_progress=True
        )

        self.assertIsInstance(volumes_info, list)
        self.assertEqual(len(volumes_info), volumes)

        with self.assertRaises(ValueError):
            volumes_info = self.volume_api.upload_dirs(
                dataset_id, self.files_path_1, progress_cb=progress_cb
            )

        with self.assertRaises(ValueError):
            volumes_info = self.volume_api.upload_dir(
                dataset_id, self.files_path_2, progress_cb=progress_cb
            )

        volumes_info = self.volume_api.upload_dirs(
            dataset_id, [self.files_path_1], progress_cb=progress_cb
        )
        self.assertEqual(len(volumes_info), volumes)


if __name__ == "__main__":
    unittest.main()
