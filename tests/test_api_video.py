import unittest
from unittest.mock import patch

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectType
from supervisely.video.video import UnsupportedVideoFormat


class TestVideoApi(unittest.TestCase):
    workspace_id = None
    project_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        workspace_id = input("Enter workspace ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            cls.workspace_id = int(workspace_id)

    def setUp(self):
        self.project_api = self.api.project
        self.dataset_api = self.api.dataset
        self.video_api = self.api.video
        self.valid_name = "valid.mp4"
        self.invalid_name_wo_ext = "invalid"
        self.invalid_name_ext = "invalid.avi"
        # replace with paths for existing files
        self.path_valid_ext = "/video.mp4"
        self.path_invalid_ext = "/video.png"

    def tearDown(self):
        self.api.project.remove(self.project_id)

    def create_test_projects(self, count):
        """
        Create test projects for testing.
        """
        created_project_ids = []
        for i in range(count):
            created_project = self.project_api.create(
                self.workspace_id,
                name=f"[UT] Video Project {i+1}",
                type=ProjectType.VIDEOS,
                change_name_if_conflict=True,
            )
            created_project_ids.append(created_project.id)
        return created_project_ids

    def create_test_datasets(self, count):
        """
        Create test datasets for testing.
        """
        created_dataset_ids = []
        for i in range(count):
            created_dataset = self.dataset_api.create(
                self.project_id,
                name=f"[UT] Dataset {i+1}",
                change_name_if_conflict=True,
            )
            created_dataset_ids.append(created_dataset.id)
        return created_dataset_ids

    def test_upload_paths(self):
        self.project_id = self.create_test_projects(1)[0]
        created_datase_id = self.create_test_datasets(1)[0]
        # Verify if extension is supported (file path)
        with self.assertRaises(UnsupportedVideoFormat):
            self.video_api.upload_paths(
                created_datase_id, [self.invalid_name_wo_ext], [self.path_invalid_ext]
            )
        # Verify if extensions in the file name and path match
        with self.assertRaises(ValueError):
            self.video_api.upload_paths(
                created_datase_id, [self.invalid_name_ext], [self.path_valid_ext]
            )
        self.assertIsNotNone(
            self.video_api.upload_paths(created_datase_id, [self.valid_name], [self.path_valid_ext])
        )

    def test_upload_links_with_valid_links(self):
        self.project_id = self.create_test_projects(1)[0]
        created_datase_id = self.create_test_datasets(1)[0]
        video_info = self.video_api.upload_paths(
            created_datase_id, [self.valid_name], [self.path_valid_ext]
        )[0]
        hash = video_info.hash
        name = "new_" + self.valid_name
        link_video_info = self.video_api.upload_hashes(created_datase_id, [name], [hash])[0]
        self.assertIsNotNone(link_video_info)
        with self.assertRaises(UnsupportedVideoFormat):
            self.video_api.upload_hashes(created_datase_id, [self.invalid_name_wo_ext], [hash])


if __name__ == "__main__":
    unittest.main()
