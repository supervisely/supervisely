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
from supervisely.api.image_api import ImageInfo
from supervisely.io.fs import get_file_name_with_ext, list_files, list_files_recursively


class TestImageApi(unittest.TestCase):
    project_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        workspace_id = input("Enter workspace ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            project = cls.api.project.create(
                workspace_id=int(workspace_id),
                name="[UT] Image api",
                change_name_if_conflict=True,
            )
        cls.project_id = project.id

    @classmethod
    def tearDownClass(cls):
        cls.api.project.remove(cls.project_id)

    def setUp(self):
        # Set up any necessary test data or configurations
        self.image_api = self.api.image
        self.dataset_instance = self.api.dataset
        self.files_path = "/test_assets/images_2"
        self.files_path_2 = "/test_assets/images_3"
        self.files_path_3 = "/test_assets/images"
        self.all_paths = [self.files_path, self.files_path_2, self.files_path_3]

    def create_test_datasets(self, count):
        """
        Create test datasets for testing.
        """
        created_dataset_ids = []
        for i in range(count):
            created_dataset = self.dataset_instance.create(
                self.project_id,
                name=f"[UT] Dataset image {i+1}",
                change_name_if_conflict=True,
            )
            created_dataset_ids.append(created_dataset.id)
        return created_dataset_ids

    def test_upload_paths(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        image_paths = list_files(self.files_path, filter_fn=sly.image.is_valid_format)
        names = [get_file_name_with_ext(image_path) for image_path in image_paths]
        progress_cb = tqdm(total=len(image_paths), desc="Uploading images")
        # progress_cb = sly.tqdm_sly(total=len(image_paths), desc="Uploading images")
        # progress_cb = sly.Progress(total_cnt=len(image_paths), message="Uploading images").iters_done_report

        # Call the method being tested
        images_info = self.image_api.upload_paths(dataset_id, names, image_paths, progress_cb)

        # Verify the method returns the correct value
        self.assertIsInstance(images_info, list)
        self.assertEqual(len(images_info), len(image_paths))
        for info, name in zip(images_info, names):
            self.assertIsInstance(info, ImageInfo)
            self.assertIsInstance(info.id, int)
            self.assertEqual(info.dataset_id, dataset_id)
            self.assertEqual(info.name, name)

    def test_upload_paths_duplicate_names(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        image_paths = list_files(self.files_path, filter_fn=sly.image.is_valid_format)
        names = [get_file_name_with_ext(image_path) for image_path in image_paths]
        progress_cb = tqdm(total=len(image_paths), desc="Uploading images")

        # Call the method being tested
        self.image_api.upload_paths(
            dataset_id, names, image_paths, progress_cb, change_name_if_conflict=True
        )

        # Verify the method raises error
        with self.assertRaises(ValueError):
            self.image_api.upload_paths(dataset_id, names, image_paths, progress_cb)

        images_info = self.image_api.upload_paths(
            dataset_id, names, image_paths, progress_cb, change_name_if_conflict=True
        )
        self.assertEqual(len(images_info), len(image_paths))

    def test_upload_dir(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        listed_images = list_files_recursively(
            self.files_path_2, filter_fn=sly.image.is_valid_format
        )

        progress_cb = tqdm(total=len(listed_images), desc="Uploading images")

        # Call the method being tested
        images_info = self.image_api.upload_dir(
            dataset_id,
            self.files_path_2,
            progress_cb,
            include_subdirs=True,
            change_name_if_conflict=True,
        )

        # Verify the method returns the correct value
        self.assertIsInstance(images_info, list)

        with self.assertRaises(ValueError):
            self.image_api.upload_dir(dataset_id, self.files_path_2, include_subdirs=True)

        images_info = self.image_api.upload_dir(
            dataset_id,
            self.files_path_2,
            progress_cb,
            include_subdirs=True,
            change_name_if_conflict=True,
        )
        self.assertEqual(len(images_info), len(listed_images))

    def test_upload_dirs(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        all_images = []
        for path in self.all_paths:
            listed_images = list_files_recursively(path, filter_fn=sly.image.is_valid_format)
            all_images.extend(listed_images)

        progress_cb = tqdm(total=len(all_images), desc="Uploading images")

        # Call the method being tested
        images_info = self.image_api.upload_dirs(
            self.all_paths,
            dataset_id,
            progress_cb,
            include_subdirs=True,
            change_name_if_conflict=True,
        )
        # Verify the method returns the correct value
        self.assertIsInstance(images_info, list)
        self.assertEqual(len(images_info), len(all_images))

        with self.assertRaises(ValueError):
            self.image_api.upload_dirs(self.all_paths, dataset_id, include_subdirs=True)


if __name__ == "__main__":
    unittest.main()
