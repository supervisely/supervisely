from __future__ import annotations

import uuid
from typing import Dict, Optional, Tuple, Union

from supervisely._utils import take_with_default
from supervisely.annotation.tag_meta import TagMeta
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.api.module_api import ApiField
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_tag import VideoTag


class PointcloudEpisodeTag(VideoTag):
    """
    PointcloudEpisodeTag object for :class:`PointcloudEpisodeAnnotation<supervisely.pointcloud_annotation.pointcloud_episode_annotation.PointcloudEpisodeAnnotation>`. :class:`PointcloudEpisodeTag<PointcloudEpisodeTag>` object is immutable.

    :param meta: General information about point cloud episodes Tag.
    :type meta: TagMeta
    :param value: point cloud episodes Tag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
    :type value: Optional[Union[str, int, float]]
    :param frame_range: point cloud episodes Tag frame range.
    :type frame_range: Tuple[int, int] or List[int, int], optional
    :param key: uuid.UUID object.
    :type key: uuid.UUID, optional
    :param sly_id: PointcloudEpisodeTag ID in Supervisely.
    :type sly_id: int, optional
    :param labeler_login: Login of user who created PointcloudEpisodeTag.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when PointcloudEpisodeTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when PointcloudEpisodeTag was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :param is_finished: Pointcloud Episode Tag is finished or not (applicable for range tags).
    :type is_finished: bool, optional
    :param non_final_value: Pointcloud Episode Tag value is final or not. Can be useful to create tag without value.
    :type non_final_value: bool, optional
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        meta_car = sly.TagMeta('car', sly.TagValueType.NONE)
        # Now we can create a PointcloudEpisodeTag using our TagMeta
        tag_car = sly.PointcloudEpisodeTag(meta_car)
        # When you are creating a new Tag
        # Tag.value is automatically cross-checked against your TagMeta value type to make sure the value is valid.
        # If we now try to add a value to our newly created Tag, we receive "ValueError", because our TagMeta value type is "NONE"
        tag_car = sly.PointcloudEpisodeTag(meta_car, value="Bus")
        # Output: ValueError: Tag car can not have value Bus

        # Let's create another Tag with a string value type and frame range
        meta_car = sly.TagMeta('cat', sly.TagValueType.ANY_STRING)
        tag_car = sly.PointcloudEpisodeTag(meta_car, value="red", frame_range=(5, 10))

        # Now let's create a Tag using TagMeta with "ONEOF_STRING" value type
        # In order to use "oneof_string value type", you must initialize a variable with possible values(see class TagMeta for more information)
        colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
        meta_car_color = sly.TagMeta('car color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
        tag_car_color = sly.PointcloudEpisodeTag(meta_car_color, value="white", frame_range=(15, 20))

        # If given value is not in a list of possible Tags, ValueError will be raised
        tag_car_color = sly.PointcloudEpisodeTag(meta_car_color, value="yellow")
        # Output: ValueError: Tag car color can not have value yellow
    """

    _SUPPORT_UNFINISHED_TAGS = True

    def __init__(
        self,
        meta,
        value=None,
        frame_range=None,
        key=None,
        sly_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
        is_finished=None,
        non_final_value=None,
    ):
        super().__init__(
            meta,
            value,
            frame_range,
            key,
            sly_id,
            labeler_login,
            updated_at,
            created_at,
            is_finished=is_finished,
            non_final_value=non_final_value,
        )

    @classmethod
    def from_json(
        cls,
        data: Dict,
        tag_meta_collection: TagMetaCollection,
        key_id_map: Optional[KeyIdMap] = None,
    ) -> PointcloudEpisodeTag:
        """
        Convert a json dict to PointcloudEpisodeTag. Read more about `Supervisely format <https://docs.supervisely.com/data-organization/00_ann_format_navi>`_.

        :param data: PointcloudEpisodeTag in json format as a dict.
        :type data: dict
        :param tag_meta_collection: :class:`TagMetaCollection<supervisely.annotation.tag_meta_collection.TagMetaCollection>` object.
        :type tag_meta_collection: TagMetaCollection
        :param key_id_map: Key ID Map object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudEpisodeTag object
        :rtype: :class:`PointcloudEpisodeTag<PointcloudEpisodeTag>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            tag_car_color_json = {
                "frameRange": [15, 20],
                "key": "da9ca75e97744fc5aaf24d6be2eb2832",
                "name": "car color",
                "value": "white"
            }

            colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
            meta_car_color = sly.TagMeta('car color', sly.TagValueType.ONEOF_STRING, possible_values=colors)
            meta_car_collection = sly.TagMetaCollection([meta_car_color])

            tag_car_color = sly.PointcloudEpisodeTag.from_json(tag_car_color_json, meta_car_collection)
        """
        is_finished = data.get(ApiField.IS_FINISHED, True)
        non_final_value = data.get(ApiField.NON_FINAL_VALUE, False)
        temp = super(PointcloudEpisodeTag, cls).from_json(data, tag_meta_collection, key_id_map)

        return cls(
            meta=temp.meta,
            value=temp.value,
            frame_range=temp.frame_range,
            key=temp.key(),
            sly_id=temp.sly_id,
            labeler_login=temp.labeler_login,
            updated_at=temp.updated_at,
            created_at=temp.created_at,
            is_finished=is_finished,
            non_final_value=non_final_value,
        )

    def __eq__(self, other: PointcloudEpisodeTag) -> bool:
        return super().__eq__(other)

    def clone(
        self,
        meta: Optional[TagMeta] = None,
        value: Optional[Union[str, int, float]] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        key: Optional[uuid.UUID] = None,
        sly_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        is_finished: Optional[bool] = None,
        non_final_value: Optional[bool] = None,
    ) -> PointcloudEpisodeTag:
        """
        Makes a copy of PointcloudEpisodeTag with new fields, if fields are given, otherwise it will use fields of the original PointcloudEpisodeTag.

        :param meta: General information about PointcloudEpisodeTag.
        :type meta: TagMeta, optional
        :param value: PointcloudEpisodeTag value. Depends on :class:`TagValueType<TagValueType>` of :class:`TagMeta<TagMeta>`.
        :type value: Optional[Union[str, int, float]]
        :param frame_range: PointcloudEpisodeTag frame range.
        :type frame_range: Optional[Union[Tuple[int, int], List[int, int]]]
        :param key: uuid.UUID object.
        :type key: uuid.UUID, optional
        :param sly_id: PointcloudEpisodeTag ID in Supervisely.
        :type sly_id: int, optional
        :param labeler_login: Login of user who created PointcloudEpisodeTag.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when PointcloudEpisodeTag was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when PointcloudEpisodeTag was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        :param is_finished: Pointcloud Episode Tag is finished or not (applicable for range tags).
        :type is_finished: bool, optional
        :param non_final_value: Pointcloud Episode Tag value is final or not. Can be useful to create tag without value.
        :type non_final_value: bool, optional
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            colors = ["brown", "white", "black", "red", "chocolate", "gold", "grey"]
            meta_car_color = sly.TagMeta('car color', sly.TagValueType.ONEOF_STRING, possible_values=colors)

            tag_car_color = sly.PointcloudEpisodeTag(meta_car_color, value="white", frame_range=(15, 20))


            meta_bus = sly.TagMeta('bus', sly.TagValueType.ANY_STRING)

            new_tag = tag_car_color.clone(meta=meta_bus, frame_range=(15, 30), key=tag_car_color.key())
            print(new_tag.to_json())
            # Output: {
            #     "frameRange": [15, 30],
            #     "key": "4360b25778144141aa4f1a0d775a0a7a",
            #     "name": "bus",
            #     "value": "white"
            # }
        """

        return self.__class__(
            meta=take_with_default(meta, self.meta),
            value=take_with_default(value, self.value),
            frame_range=take_with_default(frame_range, self.frame_range),
            key=take_with_default(key, self.key()),
            sly_id=take_with_default(sly_id, self.sly_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
            is_finished=take_with_default(is_finished, self.is_finished),
            non_final_value=take_with_default(non_final_value, self.non_final_value),
        )
