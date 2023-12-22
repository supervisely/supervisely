# coding: utf-8

from supervisely.collection.str_enum import StrEnum


class ProjectSettings:
    def __new__(
        cls, is_group: bool = False, group_tag_id: int = None, group_images_sync: bool = None
    ):
        instance = super().__new__(cls)
        instance.is_group = is_group
        instance.group_tag_id = group_tag_id
        instance.group_images_sync = group_images_sync

        if is_group:
            if group_tag_id is None or group_images_sync is None:
                raise ValueError(
                    "When is_group is True, the values of group_tag_id and group_images_sync should be defined."
                )
            # TODO: You can add logic for group_tag_name here if needed.

        return instance.__dict__
