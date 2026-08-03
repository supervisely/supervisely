import os
import sys
import unittest
from unittest.mock import Mock

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

from supervisely.api.api import Api


class TestRemoveTeamPermanently(unittest.TestCase):
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()

    def setUp(self):
        self.team_instance = self.api.team

    def create_test_teams(self, count):
        """
        Create and archive test teams for testing 'remove_permanently'.
        """
        created_team_ids = []
        for i in range(count):
            created_team = self.team_instance.create(
                name=f"[UT] Team remove permanently {i+1}",
                change_name_if_conflict=True,
            )
            self.team_instance.archive(created_team.id)
            created_team_ids.append(created_team.id)
        return created_team_ids

    def test_single_id_deletion(self):
        # Testing deletion of a single team by ID
        created_team_ids = self.create_test_teams(1)
        single_id = created_team_ids[0]
        response = self.team_instance.remove_permanently(single_id)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for a single team deletion
        self.assertIn("taskId", response[0])
        # Verifying the team is really deleted
        team_info = self.api.team.get_info_by_id(single_id)
        self.assertIsNone(team_info)

    def test_multiple_ids_deletion(self):
        # Testing deletion of multiple teams by a list of IDs
        multiple_ids = self.create_test_teams(3)
        response = self.team_instance.remove_permanently(multiple_ids)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying the response for multiple teams deletion
        for resp in response:
            self.assertIn("taskId", resp)

    def test_batch_size_adjustment(self):
        # Testing batch size adjustment
        batch_size = 60
        created_team_ids = self.create_test_teams(1)
        single_id = created_team_ids[0]
        response = self.team_instance.remove_permanently(single_id, batch_size=batch_size)
        # Checking for a response
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        # Verifying that the batch size was correctly adjusted
        self.assertLessEqual(len(response[0]), 50)

    def test_progress_callback(self):
        # Testing the progress callback
        mock_callback = Mock()
        team_ids = self.create_test_teams(5)
        self.team_instance.remove_permanently(team_ids, progress_cb=mock_callback)
        # Verifying the callback invocation for each team
        mock_callback.assert_called_with(len(team_ids))

    def test_multiple_responses(self):
        # Testing deletion of multiple teams by a list of IDs in batch size of 1
        multiple_ids = self.create_test_teams(3)
        response = self.team_instance.remove_permanently(multiple_ids, 1)
        # Checking for a responses
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), len(multiple_ids))
        # Verifying the response for multiple teams deletion
        for resp in response:
            self.assertIn("taskId", resp)

    # TODO test that not archived teams are rejected by the server


if __name__ == "__main__":
    unittest.main()
