# isort:skip_file
import numpy as np
import os
import sys
import unittest

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)
from unittest.mock import patch

import supervisely as sly

from supervisely.api.api import Api
from supervisely.api.file_api import FileInfo


class TestStorageApi(unittest.TestCase):
    team_id = None
    api = None
    temp_dir = "test_assets/temp/"
    team_files_path = "/test_assets/test_storage_api/"
    s3_path = "s3://remote-img-test/test_img-2/"
    azure_path = "azure://supervisely-test/test_img/"
    google_path = "google://sly-dev-test/test_img/"

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        team_id = input("Enter team ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            cls.team_id = int(team_id)
        if not os.path.exists(cls.temp_dir):
            sly.fs.mkdir(cls.temp_dir)
        subdir = os.path.join(cls.temp_dir, "ds0")
        if not os.path.exists(subdir):
            sly.fs.mkdir(subdir)
        for i in range(8):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            if i < 2:
                img_path = os.path.join(cls.temp_dir, f"img_{i}.png")
            else:
                img_path = os.path.join(subdir, f"img_{i}.png")
            sly.image.write(img_path, img)
        cls.api.file.upload_directory(cls.team_id, cls.temp_dir, cls.team_files_path)

    @classmethod
    def tearDownClass(cls):
        sly.fs.remove_dir(cls.temp_dir)

    def tearDown(self) -> None:
        sly.fs.clean_dir(self.temp_dir)

    def process_path(self, path):
        dir_exists = self.api.storage.dir_exists(self.team_id, path)
        self.assertTrue(dir_exists, f"Directory should exist: {path}")

        dir_exists = self.api.storage.dir_exists(self.team_id, path + "_not_exist")
        self.assertFalse(dir_exists, f"Directory should not exist: {path}_not_exist")

        dir_info = self.api.storage.list(self.team_id, path)
        self.assertEqual(len(dir_info), 8, f"Should be a list of 8 file infos: {path}")

        first_item = dir_info[0]
        self.assertEqual(type(first_item), FileInfo, f"Should be a list of file infos: {path}")

        file_exists = self.api.storage.exists(self.team_id, first_item.path)
        self.assertTrue(file_exists, f"File should exist: {path}")

        file_exists = self.api.storage.exists(self.team_id, first_item.path + "_not_exist")
        self.assertFalse(file_exists, f"File should not exist: {path}_not_exist")

        dir_info = self.api.storage.list(self.team_id, path, recursive=False)
        self.assertEqual(len(dir_info), 3, f"Should be a list of 3 file/folder infos: {path}")

        dir_info = self.api.storage.list(self.team_id, path, limit=2)
        self.assertEqual(len(dir_info), 2, f"Should be a list of 2 file/folder infos: {path}")

        dir_info = self.api.storage.list(self.team_id, path, recursive=False, limit=2)
        self.assertEqual(len(dir_info), 2, f"Should be a list of 2 file/folder infos: {path}")

        dir_info = self.api.storage.list(self.team_id, path, return_type="dict")
        self.assertEqual(len(dir_info), 8, f"Should be a list of 8 files: {path}")
        self.assertEqual(type(dir_info[0]), dict, "Should be a list of dicts: /tmp/test img/")

        dir_info = self.api.storage.list(self.team_id, path, recursive=False, return_type="dict")
        self.assertEqual(len(dir_info), 3, f"Should be a list of 3 file infos: {path}")
        self.assertEqual(type(dir_info[0]), dict, "Should be a list of dicts: /tmp/test img/")

        local_path = os.path.join(self.temp_dir, os.path.basename(path.rstrip("/")))
        self.api.storage.download_directory(self.team_id, path, local_path)
        self.assertTrue(os.path.exists(local_path), f"Directory should be downloaded: {path}")

        local_file_path = os.path.join(self.temp_dir, first_item.name)
        self.api.storage.download(self.team_id, first_item.path, local_file_path)
        self.assertTrue(os.path.exists(local_file_path), f"File should be downloaded: {path}")

    def test_s3(self):
        self.process_path(self.s3_path)

    def test_azure(self):
        self.process_path(self.azure_path)

    def test_google(self):
        self.process_path(self.google_path)

    def test_team_files(self):
        self.process_path(self.team_files_path)


if __name__ == "__main__":
    unittest.main()
