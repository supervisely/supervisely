# coding: utf-8

from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
from supervisely.api.module_api import ApiField
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely._utils import take_with_default
import supervisely.volume_annotation.constants as constants
from supervisely.volume_annotation.plane_info import PlaneName, get_normal
from supervisely.volume_annotation.volume_object_collection import (
    VolumeObjectCollection,
)


class VolumeFigure(VideoFigure):
    def __init__(
        self,
        volume_object,
        geometry,
        plane_name,
        slice_index,
        key=None,
        class_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        super().__init__(
            video_object=volume_object,
            geometry=geometry,
            frame_index=slice_index,
            key=key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        self._plane_name = PlaneName.validate(plane_name)
        self._slice_index = slice_index

    @property
    def volume_object(self):
        return self._video_object

    @property
    def video_object(self):
        raise NotImplementedError(
            'Property "video_object" is only available for videos'
        )

    @property
    def parent_object(self):
        return self.volume_object()

    @property
    def frame_index(self):
        raise NotImplementedError('Property "frame_index" is only available for videos')

    @property
    def slice_index(self):
        return self._slice_index

    @property
    def plane_name(self):
        return self._plane_name

    @property
    def normal(self):
        return get_normal(self.plane_name)

    def _validate_geometry_type(self):
        if (
            self.parent_object.obj_class.geometry_type != AnyGeometry
            and type(self._geometry) != ClosedSurfaceMesh
        ):
            if type(self._geometry) is not self.parent_object.obj_class.geometry_type:
                raise RuntimeError(
                    "Input geometry type {!r} != geometry type of ObjClass {}".format(
                        type(self._geometry), self.parent_object.obj_class.geometry_type
                    )
                )

    def to_json(self, key_id_map=None, save_meta=False):
        data_json = super().to_json(key_id_map, save_meta)
        if save_meta is True:
            data_json[constants.META] = {
                constants.SLICE_INDEX: self.slice_index,
                constants.PLANE_NAME: self.plane_name,
                constants.NORMAL: self.normal,
            }
        return data_json

    # @TODO: from json

    def validate_bounds(self, img_size, _auto_correct=False):
        if type(self._geometry) == ClosedSurfaceMesh:
            return
        super().validate_bounds(img_size, _auto_correct)

    def clone(
        self,
        volume_object=None,
        geometry=None,
        plane_name=None,
        slice_index=None,
        key=None,
        class_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        return self.__class__(
            self,
            volume_object=take_with_default(volume_object, self.parent_object),
            geometry=take_with_default(geometry, self.geometry),
            plane_name=take_with_default(plane_name, self.plane_name),
            slice_index=take_with_default(slice_index, self.slice_index),
            key=take_with_default(key, self._key),
            class_id=take_with_default(class_id, self.class_id),
            labeler_login=take_with_default(labeler_login, self.labeler_login),
            updated_at=take_with_default(updated_at, self.updated_at),
            created_at=take_with_default(created_at, self.created_at),
        )


class VideoFigure1:
    @classmethod
    def from_json(
        cls,
        data,
        objects: VolumeObjectCollection,
        slice_index,
        key_id_map: KeyIdMap = None,
    ):
        object_id = data.get(OBJECT_ID, None)
        object_key = None
        if OBJECT_KEY in data:
            object_key = uuid.UUID(data[OBJECT_KEY])

        if object_id is None and object_key is None:
            raise RuntimeError(
                "Figure can not be deserialized from json: object_id or object_key are not found"
            )

        if object_key is None:
            if key_id_map is None:
                raise RuntimeError("Figure can not be deserialized: key_id_map is None")
            object_key = key_id_map.get_object_key(object_id)
            if object_key is None:
                raise RuntimeError(
                    "Object with id={!r} not found in key_id_map".format(object_id)
                )

        object = objects.get(object_key)
        if object is None:
            raise RuntimeError(
                "Figure can not be deserialized: corresponding object {!r} not found in ObjectsCollection".format(
                    object_key.hex
                )
            )

        shape_str = data[ApiField.GEOMETRY_TYPE]
        geometry_json = data[ApiField.GEOMETRY]

        shape = GET_GEOMETRY_FROM_STR(shape_str)
        geometry = shape.from_json(geometry_json)

        key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_figure(key, data.get(ID, None))

        class_id = data.get(CLASS_ID, None)
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)

        return cls(
            object,
            geometry,
            frame_index,
            key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
