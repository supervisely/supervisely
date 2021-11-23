import uuid

from supervisely_lib.video_annotation.video_object import VideoObject
from supervisely_lib.annotation.label import LabelJsonFields
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.volume_annotation.constants import KEY, ID
from supervisely_lib.video_annotation.video_tag_collection import VideoTagCollection
from supervisely_lib.video_annotation.key_id_map import KeyIdMap
from supervisely_lib.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, CLASS_ID


class VolumeObject(VideoObject):
    pass
