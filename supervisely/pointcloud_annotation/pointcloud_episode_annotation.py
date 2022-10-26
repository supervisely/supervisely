# coding: utf-8
from __future__ import annotations
import uuid
import json
from typing import Optional, Dict, List

from supervisely.project.project_meta import ProjectMeta
from supervisely._utils import take_with_default
from supervisely.api.module_api import ApiField
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
from supervisely.video_annotation.constants import (
    FRAMES,
    DESCRIPTION,
    FRAMES_COUNT,
    TAGS,
    OBJECTS,
    KEY,
)
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_episode_frame_collection import (
    PointcloudEpisodeFrameCollection,
)
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)


class PointcloudEpisodeAnnotation:
    def __init__(
        self,
        frames_count: Optional[int] = None,
        objects: Optional[PointcloudObjectCollection] = None,
        frames: Optional[PointcloudEpisodeFrameCollection] = None,
        tags: Optional[PointcloudEpisodeTagCollection] = None,
        description: Optional[str] = "",
        key: uuid.UUID = None,
    ) -> None:
        self._frames_count = frames_count
        self._description = description
        self._frames = take_with_default(frames, PointcloudEpisodeFrameCollection())
        self._tags = take_with_default(tags, PointcloudEpisodeTagCollection())
        self._objects = take_with_default(objects, PointcloudObjectCollection())
        self._key = take_with_default(key, uuid.uuid4())

    def get_tags_on_frame(self, frame_index: int) -> PointcloudEpisodeTagCollection:
        frame = self._frames.get(frame_index, None)
        if frame is None:
            if frame_index < self.frames_count:
                return PointcloudEpisodeTagCollection([])
            else:
                raise ValueError(f"No frame with index {frame_index} in annotation.")
        tags = []
        for tag in self._tags:
            if frame_index >= tag.frame_range[0] and frame_index <= tag.frame_range[1]:
                tags.append(tag)
        return PointcloudEpisodeTagCollection(tags)

    def get_objects_on_frame(self, frame_index: int) -> PointcloudObjectCollection:
        frame = self._frames.get(frame_index, None)
        if frame is None:
            if frame_index < self.frames_count:
                return PointcloudObjectCollection([])
            else:
                raise ValueError(f"No frame with index {frame_index} in annotation.")
        frame_objects = {}
        for fig in frame.figures:
            if fig.parent_object.key() not in frame_objects.keys():
                frame_objects[fig.parent_object.key()] = fig.parent_object
        return PointcloudObjectCollection(list(frame_objects.values()))

    def get_figures_on_frame(self, frame_index: int) -> List[PointcloudFigure]:
        frame = self._frames.get(frame_index, None)
        if frame is None:
            if frame_index < self.frames_count:
                return PointcloudObjectCollection([])
            else:
                raise ValueError(f"No frame with index {frame_index} in annotation.")
        return frame.figures

    def to_json(self, key_id_map: KeyIdMap = None) -> Dict:
        res_json = {
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FRAMES_COUNT: self.frames_count,
            FRAMES: self.frames.to_json(key_id_map),
        }

        if key_id_map is not None:
            dataset_id = key_id_map.get_video_id(self.key())
            if dataset_id is not None:
                res_json[ApiField.DATASET_ID] = dataset_id

        return res_json

    @classmethod
    def from_json(
        cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ):
        item_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(item_key, data.get(ApiField.DATASET_ID, None))

        description = data.get(DESCRIPTION, "")
        frames_count = data.get(FRAMES_COUNT, 0)

        tags = PointcloudEpisodeTagCollection.from_json(
            data[TAGS], project_meta.tag_metas, key_id_map
        )
        objects = PointcloudObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)
        frames = PointcloudEpisodeFrameCollection.from_json(
            data[FRAMES], objects, key_id_map=key_id_map
        )

        return cls(frames_count, objects, frames, tags, description, item_key)

    @classmethod
    def load_json_file(
        cls, path: str, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudEpisodeAnnotation:
        """
        Loads json file and converts it to PointcloudEpisodeAnnotation.

        :param path: Path to the json file.
        :type path: str
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudEpisodeAnnotation object
        :rtype: :class:`PointcloudEpisodeAnnotation<PointcloudEpisodeAnnotation>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            address = 'https://app.supervise.ly/'
            token = 'Your Supervisely API Token'
            api = sly.Api(address, token)

            team_name = 'Vehicle Detection'
            workspace_name = 'Cities'
            project_name =  'London'

            team = api.team.get_info_by_name(team_name)
            workspace = api.workspace.get_info_by_name(team.id, workspace_name)
            project = api.project.get_info_by_name(workspace.id, project_name)

            meta = api.project.get_meta(project.id)

            # Load json file
            path = "/home/admin/work/docs/my_dataset/ann/annotation.json"
            ann = sly.PointcloudEpisodeAnnotation.load_json_file(path, meta)
        """
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta, key_id_map)

    def clone(
        self,
        frames_count: Optional[int] = None,
        objects: Optional[PointcloudObjectCollection] = None,
        frames: Optional[PointcloudEpisodeFrameCollection] = None,
        tags: Optional[PointcloudEpisodeTagCollection] = None,
        description: Optional[str] = "",
    ) -> PointcloudEpisodeAnnotation:
        return PointcloudEpisodeAnnotation(
            frames_count=take_with_default(frames_count, self.frames_count),
            objects=take_with_default(objects, self.objects),
            frames=take_with_default(frames, self.frames),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
        )

    @property
    def frames_count(self) -> int:
        return self._frames_count

    @property
    def objects(self) -> PointcloudObjectCollection:
        return self._objects

    @property
    def frames(self) -> PointcloudEpisodeFrameCollection:
        return self._frames

    @property
    def figures(self) -> List[PointcloudFigure]:
        return self.frames.figures

    @property
    def tags(self) -> PointcloudEpisodeTagCollection:
        return self._tags

    def key(self) -> uuid.UUID:
        return self._key

    @property
    def description(self) -> str:
        return self._description

    def is_empty(self) -> bool:
        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False
