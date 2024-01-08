import unittest
from unittest.mock import Mock, patch

from supervisely.api.api import Api


class TestRemoveProjectPermanently(unittest.TestCase):
    workspace_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        workspace_id = input("Enter workspace ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            cls.workspace_id = int(workspace_id)

    def setUp(self):
        self.project_instance = self.api.project

    def create_test_projects(self, count):
        """
        Create test projects for testing 'remove_permanently'.
        """
        created_project_ids = []
        for i in range(count):
            created_project = self.project_instance.create(
                self.workspace_id,
                name=f"[UT] Project remove permanently {i+1}",
                change_name_if_conflict=True,
            )
            created_project_ids.append(created_project.id)
        return created_project_ids

    def test_single_id_deletion(self):
        # Testing deletion of a single project by ID
        created_project_ids = self.create_test_projects(1)
        single_id = created_project_ids[0]
        response = self.project_instance.remove_permanently(single_id)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for a single project deletion
        self.assertIn("success", response[0])
        # Verifying the project is really deleted
        project_info = self.api.project.get_info_by_id(single_id)
        self.assertIsNone(project_info)

    def test_multiple_ids_deletion(self):
        # Testing deletion of multiple projects by a list of IDs
        multiple_ids = self.create_test_projects(3)
        response = self.project_instance.remove_permanently(multiple_ids)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for multiple projects deletion
        for resp in response:
            self.assertIn("success", resp)

    def test_batch_size_adjustment(self):
        # Testing batch size adjustment
        batch_size = 60
        created_project_ids = self.create_test_projects(1)
        single_id = created_project_ids[0]
        response = self.project_instance.remove_permanently(single_id, batch_size=batch_size)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying that the batch size was correctly adjusted
        self.assertLessEqual(len(response[0]), 50)

    def test_progress_callback(self):
        # Testing the progress callback
        mock_callback = Mock()
        project_ids = self.create_test_projects(5)
        self.project_instance.remove_permanently(project_ids, progress_cb=mock_callback)
        # Verifying the callback invocation for each project
        mock_callback.assert_called_with(len(project_ids))

    def test_multiple_responses(self):
        # Testing deletion of multiple projects by a list of IDs in batch size of 1
        multiple_ids = self.create_test_projects(3)
        response = self.project_instance.remove_permanently(multiple_ids, 1)
        # Checking for a responses
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), len(multiple_ids))
        # Verifying the response for multiple projects deletion
        for resp in response:
            self.assertIn("success", resp)


if __name__ == "__main__":
    unittest.main()
