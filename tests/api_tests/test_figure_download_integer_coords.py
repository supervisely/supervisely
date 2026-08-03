# coding: utf-8

import os
import sys
import unittest
from unittest.mock import MagicMock

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

from supervisely.api.image_api import ImageApi  # noqa: E402


def _make_figure_api():
    api = MagicMock()
    response = MagicMock()
    response.json.return_value = {"pagesCount": 1, "entities": []}
    api.post.return_value = response
    return ImageApi(api).figure, api


class TestFigureDownloadIntegerCoords(unittest.TestCase):
    def test_integer_coords_true_by_default(self):
        figure_api, api = _make_figure_api()

        figure_api.download(dataset_id=1, image_ids=[2])

        method, data = api.post.call_args[0]
        self.assertEqual(method, "figures.list")
        self.assertIs(data["integerCoords"], True)

    def test_integer_coords_false_is_passed_through(self):
        figure_api, api = _make_figure_api()

        figure_api.download(dataset_id=1, image_ids=[2], integer_coords=False)

        _, data = api.post.call_args[0]
        self.assertIs(data["integerCoords"], False)


if __name__ == "__main__":
    unittest.main()
