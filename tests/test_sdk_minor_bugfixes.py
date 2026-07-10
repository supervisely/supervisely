import types

import pytest

from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.api.team_api import TeamApi, UsageInfo
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.video_annotation.video_tag import VideoTag
from supervisely.video_annotation.video_tag_collection import VideoTagCollection


def test_tag_from_json_can_resolve_name_via_tag_id():
    tag_meta = TagMeta("quality", TagValueType.NONE, sly_id=17)
    tag = Tag.from_json({"tagId": 17}, TagMetaCollection([tag_meta]))

    assert tag.meta.name == "quality"


def test_usage_info_from_json_reads_max_modules():
    usage = UsageInfo.from_json({"accountType": "pro", "maxModules": 12})

    assert usage == UsageInfo(plan="pro", max_modules=12)


def test_team_api_convert_json_info_uses_usage_parser():
    api = TeamApi(types.SimpleNamespace())
    team = api._convert_json_info(
        {
            "id": 1,
            "name": "Team",
            "description": "",
            "role": "admin",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
            "usage": {"accountType": "pro", "maxModules": 5},
        }
    )

    assert team.usage == UsageInfo(plan="pro", max_modules=5)


def test_video_annotation_rejects_out_of_bounds_tag_ranges():
    tag_meta = TagMeta("phase", TagValueType.NONE)
    tag = VideoTag(tag_meta, frame_range=(0, 5))

    with pytest.raises(ValueError):
        VideoAnnotation((100, 100), 5, tags=VideoTagCollection([tag]), frames=FrameCollection())
