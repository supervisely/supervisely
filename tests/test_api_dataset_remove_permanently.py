import unittest
from unittest.mock import Mock, patch

from supervisely.api.api import Api


class TestRemoveDatasetPermanently(unittest.TestCase):
    project_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        workspace_id = input("Enter workspace ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            project = cls.api.project.create(
                workspace_id=int(workspace_id),
                name="[UT] Dataset remove permanently",
                change_name_if_conflict=True,
            )
        cls.project_id = project.id

    @classmethod
    def tearDownClass(cls):
        cls.api.project.remove(cls.project_id)

    def setUp(self):
        self.dataset_instance = self.api.dataset

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

    def test_single_id_deletion(self):
        # Testing deletion of a single dataset by ID
        created_dataset_ids = self.create_test_datasets(1)
        single_id = created_dataset_ids[0]
        response = self.dataset_instance.remove_permanently(single_id)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for a single dataset deletion
        self.assertIn("success", response[0])
        # Verifying the dataset is really deleted
        dataset_info = self.api.dataset.get_info_by_id(single_id)
        self.assertIsNone(dataset_info)

    def test_multiple_ids_deletion(self):
        # Testing deletion of multiple datasets by a list of IDs
        multiple_ids = self.create_test_datasets(3)
        response = self.dataset_instance.remove_permanently(multiple_ids)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for multiple datasets deletion
        for resp in response:
            self.assertIn("success", resp)

    def test_batch_size_adjustment(self):
        # Testing batch size adjustment
        batch_size = 60
        created_dataset_ids = self.create_test_datasets(1)
        single_id = created_dataset_ids[0]
        response = self.dataset_instance.remove_permanently(single_id, batch_size=batch_size)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying that the batch size was correctly adjusted
        self.assertLessEqual(len(response[0]), 50)

    def test_progress_callback(self):
        # Testing the progress callback
        mock_callback = Mock()
        dataset_ids = self.create_test_datasets(5)
        self.dataset_instance.remove_permanently(dataset_ids, progress_cb=mock_callback)
        # Verifying the callback invocation for each dataset
        mock_callback.assert_called_with(len(dataset_ids))

    def test_multiple_responses(self):
        # Testing deletion of multiple datasets by a list of IDs in batch size of 1
        multiple_ids = self.create_test_datasets(3)
        response = self.dataset_instance.remove_permanently(multiple_ids, 1)
        # Checking for a responses
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), len(multiple_ids))
        # Verifying the response for multiple datasets deletion
        for resp in response:
            self.assertIn("success", resp)


if __name__ == "__main__":
    unittest.main()
