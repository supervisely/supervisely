# coding: utf-8

from copy import deepcopy
from re import L
import uuid
from supervisely.volume_annotation.volume_figure import VolumeFigure

from supervisely._utils import take_with_default
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.slice import Slice
from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection
from supervisely.volume_annotation.volume_object_collection import (
    VolumeObjectCollection,
)
from supervisely.volume_annotation.plane import Plane
from supervisely.volume_annotation.constants import (
    NAME,
    TAGS,
    OBJECTS,
    KEY,
    VOLUME_ID,
    VOLUME_META,
    PLANES,
    SPATIAL_FIGURES,
)


class VolumeAnnotation:
    def __init__(
        self,
        volume_meta,
        objects=None,
        plane_sagittal=None,
        plane_coronal=None,
        plane_axial=None,
        tags=None,
        spatial_figures=None,
        key=None,
    ):
        self._volume_meta = volume_meta
        self._tags = take_with_default(tags, VolumeTagCollection())
        self._objects = take_with_default(objects, VolumeObjectCollection())
        self._key = take_with_default(key, uuid.uuid4())

        self._plane_sagittal = take_with_default(
            plane_sagittal,
            Plane(Plane.SAGITTAL, volume_meta=volume_meta),
        )
        self._plane_coronal = take_with_default(
            plane_coronal,
            Plane(Plane.CORONAL, volume_meta=volume_meta),
        )
        self._plane_axial = take_with_default(
            plane_axial,
            Plane(Plane.AXIAL, volume_meta=volume_meta),
        )

        self._spatial_figures = take_with_default(spatial_figures, [])
        self.validate_figures_bounds()

    @property
    def volume_meta(self):
        return deepcopy(self._volume_meta)

    @property
    def plane_sagittal(self):
        return self._plane_sagittal

    @property
    def plane_coronal(self):
        return self._plane_coronal

    @property
    def plane_axial(self):
        return self._plane_axial

    @property
    def objects(self):
        return self._objects

    @property
    def tags(self):
        return self._tags

    @property
    def spatial_figures(self):
        return self._spatial_figures

    @property
    def figures(self):
        all_figures = []
        for plane in [self.plane_sagittal, self.plane_coronal, self.plane_axial]:
            all_figures.extend(plane.figures)
        return all_figures

    def key(self):
        return self._key

    def validate_figures_bounds(self):
        self.plane_sagittal.validate_figures_bounds()
        self.plane_coronal.validate_figures_bounds()
        self.plane_axial.validate_figures_bounds()

    def is_empty(self):
        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False

    def clone(
        self,
        volume_meta=None,
        objects=None,
        plane_sagittal=None,
        plane_coronal=None,
        plane_axial=None,
        tags=None,
        spatial_figures=None,
    ):
        return VolumeAnnotation(
            volume_meta=take_with_default(volume_meta, self.volume_meta),
            objects=take_with_default(objects, self.objects),
            plane_sagittal=take_with_default(plane_sagittal, self.plane_sagittal),
            plane_coronal=take_with_default(plane_coronal, self.plane_coronal),
            plane_axial=take_with_default(plane_axial, self.plane_axial),
            tags=take_with_default(tags, self.tags),
            spatial_figures=take_with_default(spatial_figures, self.spatial_figures),
        )

    @classmethod
    def from_json(cls, data, project_meta, key_id_map: KeyIdMap = None):
        volume_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()
        if key_id_map is not None:
            key_id_map.add_video(volume_key, data.get(VOLUME_ID, None))

        volume_meta = data[VOLUME_META]

        tags = VolumeTagCollection.from_json(
            data[TAGS], project_meta.tag_metas, key_id_map
        )
        objects = VolumeObjectCollection.from_json(
            data[OBJECTS], project_meta, key_id_map
        )

        plane_sagittal = None
        plane_coronal = None
        plane_axial = None
        for plane_json in data[PLANES]:
            if plane_json[NAME] == Plane.SAGITTAL:
                plane_sagittal = Plane.from_json(
                    plane_json,
                    Plane.SAGITTAL,
                    objects,
                    volume_meta=volume_meta,
                    key_id_map=key_id_map,
                )
            elif plane_json[NAME] == Plane.CORONAL:
                plane_coronal = Plane.from_json(
                    plane_json,
                    Plane.CORONAL,
                    objects,
                    volume_meta=volume_meta,
                    key_id_map=key_id_map,
                )
            elif plane_json[NAME] == Plane.AXIAL:
                plane_axial = Plane.from_json(
                    plane_json,
                    Plane.AXIAL,
                    objects,
                    volume_meta=volume_meta,
                    key_id_map=key_id_map,
                )
            else:
                raise RuntimeError(f"Unknown plane name {plane_json[NAME]}")

        spatial_figures = []
        for figure_json in data.get(SPATIAL_FIGURES, []):
            figure = VolumeFigure.from_json(
                figure_json,
                objects,
                plane_name=None,
                slice_index=None,
                key_id_map=key_id_map,
            )
            spatial_figures.append(figure)

        return cls(
            volume_meta=volume_meta,
            objects=objects,
            plane_sagittal=plane_sagittal,
            plane_coronal=plane_coronal,
            plane_axial=plane_axial,
            tags=tags,
            spatial_figures=spatial_figures,
            key=volume_key,
        )

    def to_json(self, key_id_map: KeyIdMap = None):
        res_json = {
            VOLUME_META: self.volume_meta,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            PLANES: [
                self.plane_sagittal.to_json(),
                self.plane_coronal.to_json(),
                self.plane_axial.to_json(),
            ],
            SPATIAL_FIGURES: [
                figure.to_json(key_id_map) for figure in self.spatial_figures
            ],
        }

        if key_id_map is not None:
            volume_id = key_id_map.get_video_id(self.key())
            if volume_id is not None:
                res_json[VOLUME_ID] = volume_id

        return res_json
