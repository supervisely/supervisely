import unittest
from unittest.mock import Mock, patch

from supervisely.api.api import Api


class TestArchiveProject(unittest.TestCase):
    workspace_id = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        workspace_id = input("Enter workspace ID for tests here >> ")
        with patch("builtins.input", return_value="user_input_value"):
            cls.workspace_id = int(workspace_id)

    def tearDown(self):
        self.api.project.remove_batch(self.project_ids)

    def setUp(self):
        self.project_instance = self.api.project
        self.url = "https://www.dropbox.com/"
        self.project_ids = []

    def create_test_projects(self, count):
        """
        Create test projects for testing 'remove_permanently'.
        """
        created_project_ids = []
        for i in range(count):
            created_project = self.project_instance.create(
                self.workspace_id,
                name=f"[UT] Archived project {i+1}",
                change_name_if_conflict=True,
            )
            created_project_ids.append(created_project.id)
        return created_project_ids

    def test_single_id_archiving(self):
        # Testing archiving of a single project
        created_project_ids = self.create_test_projects(1)
        single_id = created_project_ids[0]
        self.project_instance.archive(single_id, self.url, self.url)
        # Verifying the project is archived
        project_info = self.api.project.get_info_by_id(single_id)
        self.assertIsNotNone(project_info)
        self.assertIsNotNone(project_info.backup_archive)
        self.assertIsNotNone(project_info.backup_archive.get("url"))
        self.assertIsNotNone(project_info.backup_archive.get("annotationsUrl"))
        self.assertIsInstance(project_info.backup_archive.get("url"), str)
        self.assertIsInstance(project_info.backup_archive.get("annotationsUrl"), str)
        self.project_ids.append(single_id)

    def test_single_id_archiving_files_url(self):
        # Testing archiving of a single project only with files URL
        created_project_ids = self.create_test_projects(1)
        single_id = created_project_ids[0]
        self.project_instance.archive(single_id, self.url)
        # Verifying the project is archived
        project_info = self.api.project.get_info_by_id(single_id)
        self.assertIsNotNone(project_info)
        self.assertIsNotNone(project_info.backup_archive)
        self.assertIsNotNone(project_info.backup_archive.get("url"))
        self.assertIsInstance(project_info.backup_archive.get("url"), str)
        self.assertIsNone(project_info.backup_archive.get("annotationsUrl"))
        self.project_ids.append(single_id)

    def test_single_id_archiving_ann_url(self):
        # Testing archiving of a single project only with annotations URL
        created_project_ids = self.create_test_projects(1)
        single_id = created_project_ids[0]
        self.assertRaises(
            TypeError, lambda: self.project_instance.archive(single_id, ann_archive_url=self.url)
        )
        self.project_ids.append(single_id)

    def test_multiple_ids_archiving(self):
        # Testing archiving of multiple projects
        multiple_ids = self.create_test_projects(3)
        multiple_urls = [self.url for _ in range(3)]
        self.project_instance.archive_batch(multiple_ids, multiple_urls, multiple_urls)
        # Verifying the projects are archived
        for id in multiple_ids:
            project_info = self.api.project.get_info_by_id(id)
            self.assertIsNotNone(project_info)
            self.assertIsNotNone(project_info.backup_archive)
            self.assertIsNotNone(project_info.backup_archive.get("url"))
            self.assertIsNotNone(project_info.backup_archive.get("annotationsUrl"))
            self.assertIsInstance(project_info.backup_archive.get("url"), str)
            self.assertIsInstance(project_info.backup_archive.get("annotationsUrl"), str)
            self.project_ids.append(id)


if __name__ == "__main__":
    unittest.main()
