import os
import sys
import unittest
from unittest.mock import Mock, patch

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

from supervisely.api.api import Api


class TestRemoveWorkspacePermanently(unittest.TestCase):
    team_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        team_id = input("Enter team ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            cls.team_id = int(team_id)

    def setUp(self):
        self.workspace_instance = self.api.workspace

    def create_test_workspaces(self, count):
        """
        Create and archive test workspaces for testing 'remove_permanently'.
        """
        created_workspace_ids = []
        for i in range(count):
            created_workspace = self.workspace_instance.create(
                self.team_id,
                name=f"[UT] Workspace remove permanently {i+1}",
                change_name_if_conflict=True,
            )
            self.workspace_instance.archive(created_workspace.id)
            created_workspace_ids.append(created_workspace.id)
        return created_workspace_ids

    def test_single_id_deletion(self):
        # Testing deletion of a single workspace by ID
        created_workspace_ids = self.create_test_workspaces(1)
        single_id = created_workspace_ids[0]
        response = self.workspace_instance.remove_permanently(single_id)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for a single workspace deletion
        self.assertIn("taskId", response[0])
        # Verifying the workspace is really deleted
        workspace_info = self.api.workspace.get_info_by_id(single_id)
        self.assertIsNone(workspace_info)

    def test_multiple_ids_deletion(self):
        # Testing deletion of multiple workspaces by a list of IDs
        multiple_ids = self.create_test_workspaces(3)
        response = self.workspace_instance.remove_permanently(multiple_ids)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for multiple workspaces deletion
        for resp in response:
            self.assertIn("taskId", resp)

    def test_batch_size_adjustment(self):
        # Testing batch size adjustment
        batch_size = 60
        created_workspace_ids = self.create_test_workspaces(1)
        single_id = created_workspace_ids[0]
        response = self.workspace_instance.remove_permanently(single_id, batch_size=batch_size)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying that the batch size was correctly adjusted
        self.assertLessEqual(len(response[0]), 50)

    def test_progress_callback(self):
        # Testing the progress callback
        mock_callback = Mock()
        workspace_ids = self.create_test_workspaces(5)
        self.workspace_instance.remove_permanently(workspace_ids, progress_cb=mock_callback)
        # Verifying the callback invocation for each workspace
        mock_callback.assert_called_with(len(workspace_ids))

    def test_multiple_responses(self):
        # Testing deletion of multiple workspaces by a list of IDs in batch size of 1
        multiple_ids = self.create_test_workspaces(3)
        response = self.workspace_instance.remove_permanently(multiple_ids, 1)
        # Checking for a responses
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), len(multiple_ids))
        # Verifying the response for multiple workspaces deletion
        for resp in response:
            self.assertIn("taskId", resp)

    # TODO test that not archived workspaces are rejected by the server


if __name__ == "__main__":
    unittest.main()
