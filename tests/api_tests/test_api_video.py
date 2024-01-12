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
from supervisely.api.video.video_api import VideoInfo
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
                name="[UT] Video api",
                type=sly.ProjectType.VIDEOS,
                change_name_if_conflict=True,
            )
        cls.project_id = project.id

    @classmethod
    def tearDownClass(cls):
        cls.api.project.remove(cls.project_id)

    def setUp(self):
        # Set up any necessary test data or configurations
        self.video_api = self.api.video
        self.dataset_instance = self.api.dataset
        self.files_path = "/test_assets/videos"
        self.files_path_2 = "/test_assets/videos_2"
        self.files_path_3 = "/test_assets/videos_3"
        self.all_paths = [self.files_path, self.files_path_2, self.files_path_3]

    def create_test_datasets(self, count):
        """
        Create test datasets.
        """
        created_dataset_ids = []
        for i in range(count):
            created_dataset = self.dataset_instance.create(
                self.project_id,
                name=f"[UT] Dataset video {i+1}",
                change_name_if_conflict=True,
            )
            created_dataset_ids.append(created_dataset.id)
        return created_dataset_ids

    def test_upload_paths(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        video_paths = list_files(self.files_path, filter_fn=sly.video.is_valid_format)
        names = [get_file_name_with_ext(video_path) for video_path in video_paths]
        progress_cb = tqdm(total=len(video_paths), desc="Uploading videos")

        # Call the method being tested
        video_info = self.video_api.upload_paths(dataset_id, names, video_paths, progress_cb)

        # Verify the method returns the correct value
        self.assertIsInstance(video_info, list)
        self.assertEqual(len(video_info), len(video_paths))
        for info, name in zip(video_info, names):
            self.assertIsInstance(info, VideoInfo)
            self.assertIsInstance(info.id, int)
            self.assertEqual(info.dataset_id, dataset_id)
            self.assertEqual(info.name, name)

    def test_upload_paths_duplicate_names(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        video_paths = list_files(self.files_path, filter_fn=sly.video.is_valid_format)
        names = [get_file_name_with_ext(video_path) for video_path in video_paths]
        progress_cb = tqdm(total=len(video_paths), desc="Uploading videos")

        # Call the method being tested
        self.video_api.upload_paths(dataset_id, names, video_paths, progress_cb)

        # Verify the method raises error
        with self.assertRaises(ValueError):
            self.video_api.upload_paths(dataset_id, names, video_paths, progress_cb)

        videos_info = self.video_api.upload_paths(
            dataset_id, names, video_paths, progress_cb, change_name_if_conflict=True
        )
        self.assertEqual(len(videos_info), len(video_paths))

    def test_upload_dir(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        listed_videos = list_files(self.files_path, filter_fn=sly.video.is_valid_format)

        progress_cb = tqdm(total=len(listed_videos), desc="Uploading videos")

        # Call the method being tested
        videos_info = self.video_api.upload_dir(dataset_id, self.files_path, progress_cb)

        # Verify the method returns the correct value
        self.assertIsInstance(videos_info, list)

        with self.assertRaises(ValueError):
            self.video_api.upload_dir(dataset_id, self.files_path)

        progress_cb_1 = tqdm(total=len(listed_videos), desc="Uploading videos")
        videos_info = self.video_api.upload_dir(
            dataset_id, self.files_path, progress_cb_1, change_name_if_conflict=True
        )
        self.assertEqual(len(videos_info), len(listed_videos))

    def test_upload_dirs(self):
        # Define test data
        dataset_id = self.create_test_datasets(1)[0]
        all_videos = []
        for path in self.all_paths:
            listed_videos = list_files_recursively(path, filter_fn=sly.video.is_valid_format)
            all_videos.extend(listed_videos)

        progress_cb = tqdm(total=len(all_videos), desc="Uploading videos")

        # Call the method being tested
        videos_info = self.video_api.upload_dirs(
            self.all_paths,
            dataset_id,
            progress_cb,
            include_subdirs=True,
            change_name_if_conflict=True,
        )
        # Verify the method returns the correct value
        self.assertIsInstance(videos_info, list)
        self.assertEqual(len(videos_info), len(all_videos))

        with self.assertRaises(ValueError):
            self.video_api.upload_dirs(
                self.all_paths,
                dataset_id,
                progress_cb,
                include_subdirs=True,
            )


if __name__ == "__main__":
    unittest.main()
