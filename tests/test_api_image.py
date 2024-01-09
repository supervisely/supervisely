# isort:skip_file
import unittest
from unittest.mock import patch
import supervisely as sly  # isort:skip
from tqdm import tqdm

from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo


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
                name="[UT] Dataset image",
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
        self.files_path = "/home/ganpoweird/Work/for_test/images_2"

    def create_test_datasets(self, count):
        """
        Create test datasets for testing 'remove_permanently'.
        """
        created_dataset_ids = []
        for i in range(count):
            created_dataset = self.dataset_instance.create(
                self.project_id,
                name=f"[UT] Dataset remove permanently {i+1}",
                change_name_if_conflict=True,
            )
            created_dataset_ids.append(created_dataset.id)
        return created_dataset_ids

    def test_upload_paths(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        image_paths = sly.fs.list_files(self.files_path)
        names = [sly.fs.get_file_name_with_ext(image_path) for image_path in image_paths]
        progress_cb = tqdm(total=len(image_paths), desc="Uploading images")

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


if __name__ == "__main__":
    unittest.main()
