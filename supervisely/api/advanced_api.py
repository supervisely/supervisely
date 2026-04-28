# coding: utf-8

# docs
from typing import Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from supervisely._utils import batched
from supervisely.api.module_api import ApiField, ModuleApiBase


class AdvancedApi(ModuleApiBase):
    """Advanced API for working with images and objects."""

    def add_tag_to_object(
        self, tag_meta_id: int, figure_id: int, value: Optional[Union[str, int]] = None
    ) -> Dict:
        """Add a tag to an object.

        :param tag_meta_id: Tag meta ID.
        :type tag_meta_id: int
        :param figure_id: Figure ID.
        :type figure_id: int
        :param value: Value of the tag. If not provided, the tag will be added without a value.
        :type value: Optional[Union[str, int]]
        :returns: Dictionary with the result.
        :rtype: dict
        """
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id}
        if value is not None:
            data[ApiField.VALUE] = value
        resp = self._api.post("object-tags.add-to-object", data)
        return resp.json()

    def remove_tag_from_object(self, tag_meta_id: int, figure_id: int, tag_id: int) -> Dict:
        """
        Remove a tag from an object.

        :param tag_meta_id: Tag meta ID.
        :type tag_meta_id: int
        :param figure_id: Figure ID.
        :type figure_id: int
        :param tag_id: Tag ID.
        :type tag_id: int
        :returns: Dictionary with the result.
        :rtype: dict
        """
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.FIGURE_ID: figure_id, ApiField.ID: tag_id}
        resp = self._api.post("object-tags.remove-from-figure", data)
        return resp.json()

    def get_object_tags(self, figure_id: int) -> Dict:
        """Get tags for an object.

        :param figure_id: Figure ID.
        :type figure_id: int
        :returns: Dictionary with the result.
        :rtype: dict
        """
        data = {ApiField.ID: figure_id}
        resp = self._api.post("figures.tags.list", data)
        return resp.json()

    def remove_tag_from_image(self, tag_meta_id: int, image_id: int, tag_id: int) -> Dict:
        """Remove a tag from an image.

        :param tag_meta_id: Tag meta ID.
        :type tag_meta_id: int
        :param image_id: Image ID.
        :type image_id: int
        :param tag_id: Tag ID.
        :type tag_id: int
        :returns: Dictionary with the result.
        :rtype: dict
        """
        data = {ApiField.TAG_ID: tag_meta_id, ApiField.IMAGE_ID: image_id, ApiField.ID: tag_id}
        resp = self._api.post("image-tags.remove-from-image", data)
        return resp.json()

    def remove_tags_from_images(
        self,
        tag_meta_ids: List[int],
        image_ids: List[int],
        progress_cb: Optional[Union[tqdm, Callable]] = None,
    ) -> None:
        """
        Remove tags from images.

        :param tag_meta_ids: List of tag meta IDs.
        :type tag_meta_ids: List[int]
        :param image_ids: List of image IDs.
        :type image_ids: List[int]
        :param progress_cb: Function for control remove progress.
        :type progress_cb: Callable
        :returns: None
        :rtype: None
        """
        for batch_ids in batched(image_ids, batch_size=100):
            data = {ApiField.TAG_IDS: tag_meta_ids, ApiField.IDS: batch_ids}
            self._api.post("image-tags.bulk.remove-from-images", data)
            if progress_cb is not None:
                progress_cb(len(batch_ids))
