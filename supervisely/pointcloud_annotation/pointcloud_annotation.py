# coding: utf-8
from __future__ import annotations
from typing import Optional, List, Dict
from supervisely.project.project_meta import ProjectMeta
from copy import deepcopy
import uuid
import json

from supervisely._utils import take_with_default
from supervisely.pointcloud_annotation.pointcloud_tag_collection import PointcloudTagCollection
from supervisely.pointcloud_annotation.constants import (
    DESCRIPTION,
    TAGS,
    OBJECTS,
    KEY,
    FIGURES,
    POINTCLOUD_ID,
)
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


class PointcloudAnnotation(VideoAnnotation):
    """
    Class for creating and using PointcloudAnnotation

    :param objects: PointcloudObjectCollection
    :param figures: List[PointcloudFigure]
    :param tags: PointcloudTagCollection
    :param description: str
    :param key: uuid class object
    """

    def __init__(
        self,
        objects: Optional[PointcloudObjectCollection] = None,
        figures: Optional[List[PointcloudFigure]] = None,
        tags: Optional[PointcloudTagCollection] = None,
        description: Optional[str] = "",
        key: Optional[uuid.UUID] = None,
    ):

        self._description = description
        self._tags = take_with_default(tags, PointcloudTagCollection())
        self._objects = take_with_default(objects, PointcloudObjectCollection())
        self._figures = take_with_default(figures, [])
        self._key = take_with_default(key, uuid.uuid4())

    @property
    def img_size(self):
        raise NotImplementedError("Not supported for pointcloud")

    @property
    def frames_count(self):
        raise NotImplementedError("Not supported for pointcloud")

    @property
    def frames(self):
        raise NotImplementedError("Not supported for pointcloud")

    @property
    def tags(self) -> PointcloudTagCollection:
        return super().tags

    @property
    def objects(self) -> PointcloudObjectCollection:
        return super().objects

    @property
    def figures(self) -> List[PointcloudFigure]:
        return deepcopy(self._figures)

    # def get_objects_on_frame(self, frame_index: int):
    #     raise NotImplementedError("Not supported for pointcloud")

    # def get_tags_on_frame(self, frame_index: int):
    #     raise NotImplementedError("Not supported for pointcloud")

    def get_objects_from_figures(self) -> PointcloudObjectCollection:
        ann_objects = {}
        for fig in self.figures:
            if fig.parent_object.key() not in ann_objects.keys():
                ann_objects[fig.parent_object.key()] = fig.parent_object

        return PointcloudObjectCollection(ann_objects.values())

    def validate_figures_bounds(self):
        raise NotImplementedError("Not supported for pointcloud")

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> Dict:
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
            FIGURES: [figure.to_json(key_id_map) for figure in self.figures],
        }

        if key_id_map is not None:
            pointcloud_id = key_id_map.get_video_id(self.key())
            if pointcloud_id is not None:
                res_json[POINTCLOUD_ID] = pointcloud_id

        return res_json

    @classmethod
    def from_json(
        cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudAnnotation:
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
        tags = PointcloudTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = PointcloudObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = PointcloudFigure.from_json(figure_json, objects, None, key_id_map)
            figures.append(figure)

        return cls(
            objects=objects, figures=figures, tags=tags, description=description, key=item_key
        )

    @classmethod
    def load_json_file(
        cls, path: str, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> PointcloudAnnotation:
        """
        Loads json file and converts it to PointcloudAnnotation.

        :param path: Path to the json file.
        :type path: str
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: PointcloudAnnotation object
        :rtype: :class:`PointcloudAnnotation<PointcloudAnnotation>`
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
            ann = sly.PointcloudAnnotation.load_json_file(path, meta)
        """
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta, key_id_map)

    def clone(
        self,
        objects: Optional[PointcloudObjectCollection] = None,
        figures: Optional[List] = None,
        tags: Optional[PointcloudTagCollection] = None,
        description: Optional[str] = None,
    ) -> PointcloudAnnotation:
        """
        :param objects: PointcloudObjectCollection
        :param figures: list of figures
        :param tags: PointcloudTagCollection
        :param description: str
        :return: PointcloudAnnotation class object
        """
        return PointcloudAnnotation(
            objects=take_with_default(objects, self.objects),
            figures=take_with_default(figures, self.figures),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
        )
