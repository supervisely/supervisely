# coding: utf-8


# docs
from typing import Optional, List, Dict
from supervisely.project.project_meta import ProjectMeta
from copy import deepcopy
import uuid

from supervisely._utils import take_with_default
from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.constants import FRAMES, IMG_SIZE, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, \
                                                       DESCRIPTION, FRAMES_COUNT, TAGS, OBJECTS, VIDEO_ID, KEY, \
                                                       VIDEOS_MAP, VIDEO_NAME
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.video_annotation.constants import FIGURES
from supervisely.pointcloud_annotation.constants import POINTCLOUD_ID
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection


class PointcloudAnnotation(VideoAnnotation):
    """
    Class for creating and using PointcloudAnnotation
    """
    def __init__(self, objects: Optional[VideoObjectCollection]=None, figures: Optional[List]=None, tags: Optional[VideoTagCollection]=None,
                 description: Optional[str]="", key: Optional[uuid.UUID]=None):
        """
        :param objects: VideoObjectCollection
        :param figures: list of figures(Point, Cuboid, etc)
        :param tags: VideoTagCollection
        :param description: str
        :param key: uuid class object
        """
        self._description = description
        self._tags = take_with_default(tags, VideoTagCollection())
        self._objects = take_with_default(objects, VideoObjectCollection())
        self._figures = take_with_default(figures, [])
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def img_size(self):
        raise RuntimeError("Not supported for pointcloud")

    @property
    def frames_count(self):
        raise RuntimeError("Not supported for pointcloud")

    @property
    def figures(self):
        return deepcopy(self._figures)

    def validate_figures_bounds(self):
        raise RuntimeError("Not supported for pointcloud")

    def to_json(self, key_id_map: Optional[KeyIdMap]=None) -> Dict:
        """
        The function to_json convert PointcloudAnnotation to json format
        :param key_id_map: KeyIdMap class object
        :return: PointcloudAnnotation in json format
        """
        res_json = {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FIGURES: [figure.to_json(key_id_map) for figure in self.figures]
        }

        if key_id_map is not None:
            pointcloud_id = key_id_map.get_video_id(self.key())
            if pointcloud_id is not None:
                res_json[POINTCLOUD_ID] = pointcloud_id

        return res_json

    @classmethod
    def from_json(cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap]=None):
        """
        :param data: input PointcloudAnnotation in json format
        :param project_meta: ProjectMeta class object
        :param key_id_map: KeyIdMap class object
        :return: PointcloudAnnotation class object
        """
        item_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()
        if key_id_map is not None:
            key_id_map.add_video(item_key, data.get(POINTCLOUD_ID, None))
        description = data.get(DESCRIPTION, "")
        tags = VideoTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = PointcloudObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = PointcloudFigure.from_json(figure_json, objects, None, key_id_map)
            figures.append(figure)

        return cls(objects=objects,
                   figures=figures,
                   tags=tags,
                   description=description,
                   key=item_key)

    def clone(self, objects: Optional[VideoObjectCollection]=None, figures: Optional[List]=None,
              tags: Optional[VideoTagCollection]=None, description: Optional[str]=None):
        """
        :param objects: VideoObjectCollection
        :param figures: list of figures
        :param tags: VideoTagCollection
        :param description: str
        :return: PointcloudAnnotation class object
        """
        return PointcloudAnnotation(objects=take_with_default(objects, self.objects),
                                    figures=take_with_default(figures, self.figures),
                                    tags=take_with_default(tags, self.tags),
                                    description=take_with_default(description, self.description))
