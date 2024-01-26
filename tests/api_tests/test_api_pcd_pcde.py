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
from supervisely.io.fs import list_files_recursively


class TestPCDApi(unittest.TestCase):
    pcd_project_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        workspace_id = input("Enter workspace ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            pcd_project = cls.api.project.create(
                workspace_id=int(workspace_id),
                name="[UT] PCD api",
                type=sly.ProjectType.POINT_CLOUDS,
                change_name_if_conflict=True,
            )
        cls.pcd_project_id = pcd_project.id
        with patch("builtins.input", return_value="user_input_value"):
            pcde_project = cls.api.project.create(
                workspace_id=int(workspace_id),
                name="[UT] PCDE api",
                type=sly.ProjectType.POINT_CLOUD_EPISODES,
                change_name_if_conflict=True,
            )
        cls.pcde_project_id = pcde_project.id

    @classmethod
    def tearDownClass(cls):
        cls.api.project.remove_permanently(cls.pcd_project_id)
        cls.api.project.remove_permanently(cls.pcde_project_id)

    def setUp(self):
        # Set up any necessary test data or configurations
        self.pcd_api = self.api.pointcloud
        self.pcde_api = self.api.pointcloud_episode
        self.dataset_instance = self.api.dataset
        self.files_path_pcd = "/test_assets/pcds/pointcloud"
        self.files_path_pcde = "/test_assets/pcde/pointcloud"
        self.pcd_dataset = self.dataset_instance.create(
            self.pcd_project_id,
            name=f"[UT] Dataset PCD",
            change_name_if_conflict=True,
        ).id
        self.pcde_dataset = self.dataset_instance.create(
            self.pcde_project_id,
            name=f"[UT] Dataset PCDE",
            change_name_if_conflict=True,
        ).id

    def test_upload_dirs_pcd(self):
        pcd_paths = list_files_recursively(self.files_path_pcd)
        progress_cb = tqdm(total=len(pcd_paths), desc="Uploading PCDs")
        pcd_info = self.pcd_api.upload_dirs(
            self.pcd_dataset, [self.files_path_pcd], progress_cb=progress_cb
        )

        self.assertIsInstance(pcd_info, list)
        self.assertEqual(len(pcd_info), len(pcd_paths))
        for info in pcd_info:
            self.assertIsInstance(info, PointcloudInfo)
            self.assertIsInstance(info.id, int)
            self.assertEqual(info.dataset_id, self.pcd_dataset)

        with self.assertRaises(ValueError):
            self.pcd_api.upload_dir(
                self.pcd_dataset, self.files_path_pcd, change_name_if_conflict=False
            )

        pcd_info = self.pcd_api.upload_dirs(
            self.pcd_dataset, [self.files_path_pcd], progress_cb=progress_cb
        )

    def test_upload_dirs_pcde(self):
        pcde_paths = list_files_recursively(self.files_path_pcde)
        progress_cb = tqdm(total=len(pcde_paths), desc="Uploading PCDEs")
        pcde_info = self.pcde_api.upload_dirs(
            self.pcde_dataset, [self.files_path_pcde], progress_cb=progress_cb
        )

        self.assertIsInstance(pcde_info, list)
        self.assertEqual(len(pcde_info), len(pcde_paths))
        for info in pcde_info:
            self.assertIsInstance(info, PointcloudInfo)
            self.assertIsInstance(info.id, int)
            self.assertEqual(info.dataset_id, self.pcde_dataset)

        with self.assertRaises(ValueError):
            self.pcde_api.upload_dir(
                self.pcde_dataset, self.files_path_pcde, change_name_if_conflict=False
            )

        pcde_info = self.pcde_api.upload_dirs(
            self.pcde_dataset, [self.files_path_pcde], progress_cb=progress_cb
        )


if __name__ == "__main__":
    unittest.main()
