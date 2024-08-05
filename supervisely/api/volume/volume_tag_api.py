# coding: utf-8
from typing import Optional, Union

from supervisely.annotation.tag_meta import TagMeta
from supervisely.api.entity_annotation.tag_api import TagApi
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap


class VolumeTagApi(TagApi):
    """
    :class:`VolumeTag<supervisely.volume_annotation.volume_tag.VolumeTag>` for a single volume. :class:`VolumeTagApi<VolumeTagApi>` object is immutable.
    """

    _entity_id_field = ApiField.ENTITY_ID
    _method_bulk_add = "volumes.tags.bulk.add"

    def remove_from_volume(self, tag_id: int):
        """
        Remove tag from volume.

        :param tag_id: VolumeTag ID in Supervisely.
        :type tag_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.volume.tag.remove_from_volume(volume_tag_id)

        """

        self._api.post("volumes.tags.remove", {ApiField.ID: tag_id})

    def update_value(self, tag_id: int, tag_value: Union[str, int]):
        """
        Update VolumeTag value.

        :param tag_id: VolumeTag ID in Supervisely.
        :type tag_id: int
        :param tag_value: New VolumeTag value.
        :type tag_value: str or int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.volume.tag.update_value(volume_tag_id, 'new_tag_value')
        """

        self._api.post(
            "volumes.tags.update-value",
            {ApiField.ID: tag_id, ApiField.VALUE: tag_value},
        )

    def append_to_volume(
        self,
        volume_id: int,
        tag_id: int,
        value: Union[str, int, None] = None,
        tag_meta: Optional[TagMeta] = None,
    ) -> int:
        """
        Add tag to volume.

        :param volume_id: Volume ID in Supervisely.
        :type volume_id: int
        :param tag_id: Tag ID in Supervisely.
        :type tag_id: int
        :param tag_value: VolumeTag value.
        :type tag_value: str or int or None, optional
        :param tag_meta: TagMeta object.
        :type tag_meta: TagMeta, optional
        :return: VolumeTag ID.
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            volume_id = 19402023
            tag_id = 19402023
            tag_value = 'tag_value'
            tag_meta = api.tag.get_info_by_id(tag_id).meta
            id = api.volume.tag.append_to_volume(volume_id, tag_id, tag_value, tag_meta)
        """
        data = {
            ApiField.ENTITY_ID: volume_id,
            ApiField.TAG_ID: tag_id,
        }

        if tag_meta:
            if not tag_meta.is_valid_value(value):
                raise ValueError("Tag {} can not have value {}".format(tag_meta.name, value))
        if value is not None:
            data[ApiField.VALUE] = value
        response = self._api.post("volumes.tags.add", data)

        return response.json().get(ApiField.ID)
