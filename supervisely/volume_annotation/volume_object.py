# coding: utf-8

import uuid
from typing import Optional, Union

from numpy import ndarray

from supervisely.geometry.mask_3d import Mask3D
from supervisely.video_annotation.video_object import VideoObject
from supervisely.volume_annotation import volume_figure
from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection


class VolumeObject(VideoObject):
    """Object in volume annotation: obj_class, optional Mask3D geometry, tags. Immutable."""

    def __init__(
        self,
        obj_class,
        tags: Optional[VolumeTagCollection] = None,
        key: Optional[uuid.UUID] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
        mask_3d: Optional[Union[Mask3D, ndarray, str]] = None,
        custom_data: Optional[dict] = None,
    ):
        """
        Object in volume annotation.

        :param obj_class: Object class (e.g. 'heart' with Rectangle).
        :type obj_class: :class:`~supervisely.annotation.obj_class.ObjClass`
        :param tags: Tags for this object.
        :type tags: :class:`~supervisely.volume_annotation.volume_tag_collection.VolumeTagCollection`, optional
        :param key: UUID key. Auto-generated if not provided.
        :type key: uuid.UUID, optional
        :param class_id: Server-side class ID.
        :type class_id: int, optional
        :param labeler_login: Login of user who created the object.
        :type labeler_login: str, optional
        :param updated_at: Last modification timestamp (ISO format).
        :type updated_at: str, optional
        :param created_at: Creation timestamp (ISO format).
        :type created_at: str, optional
        :param mask_3d: Geometry: file path, numpy array, or Mask3D. Creates VolumeFigure if set.
        :type mask_3d: Union[str, ndarray, :class:`~supervisely.geometry.mask_3d.Mask3D`], optional
        :param custom_data: Custom data for the figure.
        :type custom_data: dict, optional

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
                volume_obj_heart = sly.VolumeObject(obj_class_heart)
                print(volume_obj_heart.to_json())
        """
        super().__init__(
            obj_class=obj_class,
            tags=tags,
            key=key,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
        if mask_3d is not None:
            if isinstance(mask_3d, str):
                self.figure = volume_figure.VolumeFigure(
                    self,
                    geometry=Mask3D.create_from_file(mask_3d),
                    labeler_login=labeler_login,
                    updated_at=updated_at,
                    created_at=created_at,
                    custom_data=custom_data,
                )
            elif isinstance(mask_3d, ndarray):
                self.figure = volume_figure.VolumeFigure(
                    self,
                    geometry=Mask3D(mask_3d),
                    labeler_login=labeler_login,
                    updated_at=updated_at,
                    created_at=created_at,
                    custom_data=custom_data,
                )
            elif isinstance(mask_3d, Mask3D):
                self.figure = volume_figure.VolumeFigure(
                    self,
                    geometry=mask_3d,
                    labeler_login=labeler_login,
                    updated_at=updated_at,
                    created_at=created_at,
                    custom_data=custom_data,
                )
            else:
                raise TypeError("mask_3d type must be one of [Mask3D, ndarray, str]")
