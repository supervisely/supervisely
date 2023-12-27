# coding: utf-8

from supervisely.collection.str_enum import StrEnum


class ProjectSettings:
    def __init__(
        self,
        enable_multiview: bool = False,
        multiview_tag_name: str = None,
        views_are_synced: bool = None,
        multiview_tag_id: int = None,
    ):
        self.enable_multiview = enable_multiview
        self.multiview_tag_name = multiview_tag_name
        self.multiview_tag_id = multiview_tag_id
        self.views_are_synced = views_are_synced

        if enable_multiview is True:
            if multiview_tag_name is None and multiview_tag_id is None:
                raise ValueError(
                    "When multi-view mode is enabled, the value of multiview_tag_name or multiview_tag_id should be defined."
                )
            else:
                if views_are_synced is None:
                    raise ValueError(
                        "When multi-view mode is enabled, the value of views_are_synced should be defined."
                    )

    def to_json(self, Fields):
        return {
            Fields.MULTI_VIEW: {
                Fields.ENABLED: self.enable_multiview,
                Fields.TAG_NAME: self.multiview_tag_name,
                Fields.TAG_ID: self.multiview_tag_id,
                Fields.VIEWS_ARE_SYNCED: self.views_are_synced,
            }
        }
