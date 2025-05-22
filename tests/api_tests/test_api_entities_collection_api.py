# pylint: skip-file

import os
import sys
import time
import unittest

import numpy as np

from supervisely.api.api import Api, ApiField
from supervisely.api.entities_collection_api import (
    CollectionItem,
    CollectionType,
    CollectionTypeFilter,
)

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)


class TestEntitiesCollectionApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = Api.from_env()
        cls.entities_collection_api = cls.api.entities_collection
        cls.workspace_id = 228

        project_info = cls.api.project.create(
            cls.workspace_id, "Test Project", change_name_if_conflict=True
        )
        dataset_info = cls.api.dataset.create(
            project_info.id, "Test Dataset", change_name_if_conflict=True
        )
        # Create a simple test image (10x10 red square)
        image_np_1 = np.zeros((300, 300, 3), dtype=np.uint8)
        image_np_1[:, :, 0] = 255  # Red channel

        # Create another test image (blue square)
        image_np_2 = np.zeros((300, 300, 3), dtype=np.uint8)
        image_np_2[:, :, 2] = 255  # Blue channel

        image_np_3 = np.zeros((300, 300, 3), dtype=np.uint8)
        image_np_3[:, :, 1] = 255  # Green channel

        image_infos = cls.api.image.upload_nps(
            dataset_info.id,
            ["test_1.jpg", "test_2.jpg", "test_3.jpg"],
            [image_np_1, image_np_2, image_np_3],
        )
        # Sample data for testing
        cls.project_id = project_info.id
        cls.collection_name = "Test Collection" + f"{time.time()}"
        cls.collection_description = "Test Description"
        cls.item_id_1 = image_infos[0].id
        cls.item_id_2 = image_infos[1].id
        cls.item_id_3 = image_infos[2].id
        cls.ai_search_key = "0ed6a5256433bbe32822949d563d476a"

        # Create the collection once for all tests
        result = cls.entities_collection_api.create(
            project_id=cls.project_id,
            name=cls.collection_name,
            description=cls.collection_description,
            type=CollectionType.DEFAULT,
            ai_search_key=None,
        )
        cls.collection_id = result.id

    @classmethod
    def tearDownClass(cls):
        # Clean up all collections created during tests
        collections = cls.entities_collection_api.get_list(project_id=cls.project_id)
        print("\n")
        for collection in collections:
            try:
                cls.entities_collection_api.remove(collection.id)
                print(f"Removed collection with ID: {collection.id}")
            except Exception as e:
                print(f"Failed to remove collection with ID: {collection.id}. Error: {str(e)}")

        # Clean up the project
        print("\n")
        try:
            cls.api.project.remove(cls.project_id)
            print(f"Removed project with ID: {cls.project_id}")
        except Exception as e:
            print(f"Failed to remove project with ID: {cls.project_id}. Error: {str(e)}")

    def test_E001_info_sequence(self):
        expected_sequence = [
            ApiField.ID,
            ApiField.NAME,
            ApiField.TEAM_ID,
            ApiField.PROJECT_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.DESCRIPTION,
            ApiField.OPTIONS,
            ApiField.TYPE,
            ApiField.AI_SEARCH_KEY,
        ]
        self.assertEqual(self.entities_collection_api.info_sequence(), expected_sequence)

    def test_E002_info_tuple_name(self):
        self.assertEqual(self.entities_collection_api.info_tuple_name(), "EntitiesCollectionInfo")

    def test_E003_create_default_collection(self):
        result = self.entities_collection_api.get_info_by_id(self.collection_id)
        self.collection_id = result.id
        self.assertIsInstance(result.id, int)
        self.assertEqual(result.name, self.collection_name)
        self.assertEqual(result.project_id, self.project_id)
        self.assertEqual(result.description, self.collection_description)
        self.assertEqual(result.type, CollectionType.DEFAULT)
        self.assertIsNone(result.ai_search_key)

    def test_E004_get_info_by_id(self):
        result = self.entities_collection_api.get_info_by_id(self.collection_id)

        self.assertEqual(result.id, self.collection_id)
        self.assertEqual(result.name, self.collection_name)
        self.assertEqual(result.project_id, self.project_id)

    def test_E005_create_ai_search_collection(self):
        ai_search_name = self.collection_name + " AI Search" + f"{time.time()}"
        result = self.entities_collection_api.create(
            project_id=self.project_id,
            name=ai_search_name,
            description=self.collection_description,
            type=CollectionType.AI_SEARCH,
            ai_search_key=self.ai_search_key,
        )
        self.collection_id = result.id
        self.assertEqual(result.type, CollectionType.AI_SEARCH)

        result = self.entities_collection_api.get_info_by_id(result.id, with_meta=True)

        self.assertEqual(result.ai_search_key, self.ai_search_key)

    def test_E006_get_ai_search_collection_by_key(self):
        result = self.entities_collection_api.get_info_by_ai_search_key(
            project_id=self.project_id, ai_search_key=self.ai_search_key
        )

        self.assertEqual(result.type, CollectionType.AI_SEARCH)
        self.assertEqual(result.ai_search_key, self.ai_search_key)

    def test_E007_create_ai_search_without_key_raises_error(self):
        with self.assertRaises(ValueError):
            self.entities_collection_api.create(
                project_id=self.project_id,
                name=self.collection_name,
                type=CollectionType.AI_SEARCH,
            )

    def test_E008_get_list(self):
        result = self.entities_collection_api.get_list(
            project_id=self.project_id, collection_type=CollectionType.ALL, with_meta=True
        )

        self.assertEqual(len(result), 2)

    def test_E009_add_items(self):
        items = [CollectionItem(entity_id=self.item_id_3)]

        result = self.entities_collection_api.add_items(self.collection_id, items)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["imageId"], self.item_id_3)

    def test_E010_add_items_with_missing(self):
        items = [CollectionItem(entity_id=99999999)]

        with self.assertRaises(RuntimeError):
            self.entities_collection_api.add_items(self.collection_id, items)

    def test_E011_get_items(self):
        image_info = self.api.image.get_info_by_id(self.item_id_3)
        collection_info = self.entities_collection_api.get_info_by_id(self.collection_id)
        if collection_info.type == CollectionType.AI_SEARCH:
            search_type = CollectionTypeFilter.AI_SEARCH
        else:
            search_type = CollectionTypeFilter.DEFAULT
        result = self.entities_collection_api.get_items(self.collection_id, search_type)[0]

        self.assertEqual(result, image_info)

    def test_E012_remove_items(self):
        item_ids = [self.item_id_3]
        result = self.entities_collection_api.remove_items(self.collection_id, item_ids)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["imageId"], self.item_id_3)


if __name__ == "__main__":
    unittest.main()
