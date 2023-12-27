# coding: utf-8

from supervisely.collection.str_enum import StrEnum


class ProjectSettings:
    # def __new__(
    #     cls, is_group: bool = False, group_tag_id: int = None, group_images_sync: bool = None
    # ):
    #     instance = super().__new__(cls)
    #     instance.is_group = is_group
    #     instance.group_tag_id = group_tag_id
    #     instance.group_images_sync = group_images_sync

    #     if is_group:
    #         if group_tag_id is None or group_images_sync is None:
    #             raise ValueError(
    #                 "When is_group is True, the values of group_tag_id and group_images_sync should be defined."
    #             )
    #         # TODO: You can add logic for group_tag_name here if needed.

    #     return instance.__dict__

    def __init__(
        self,
        enable_multiview: bool = False,
        multiview_tag_id: int = None,
        views_are_synced: bool = None,
    ):
        self.enable_multiview = enable_multiview
        self.multiview_tag_id = multiview_tag_id
        self.views_are_synced = views_are_synced

        if enable_multiview is True:
            if multiview_tag_id is None or views_are_synced is None:
                raise ValueError(
                    "When is_group is True, the values of group_tag_id and group_images_sync should be defined."
                )

    def to_json(self):
        return self.__dict__
