# coding: utf-8

import uuid
from typing import Optional, Union

from numpy import ndarray

from supervisely.geometry.mask_3d import Mask3D
from supervisely.video_annotation.video_object import VideoObject
from supervisely.volume_annotation import volume_figure
from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection


class VolumeObject(VideoObject):
    """
    Object in volume annotation: obj_class, optional Mask3D geometry, tags. Immutable.

    :param obj_class: ObjClass object.
    :type obj_class: :class:`~supervisely.annotation.obj_class.ObjClass`
    :param tags: VolumeTagCollection object.
    :type tags: :class:`~supervisely.volume_annotation.volume_tag_collection.VolumeTagCollection`, optional
    :param key: The UUID key associated with the :class:`~supervisely.volume_annotation.volume_figure.VolumeFigure`.
    :type key: UUID, optional
    :param class_id: ID of ObjClass to which VolumeObject belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created :class:`~supervisely.volume_annotation.volume_object.VolumeObject`.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when VolumeObject was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when VolumeObject was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :param mask_3d: Path for local geometry file, array with geometry data or Mask3D geometry object
    :type mask_3d: Union[str, ndarray, :class:`~supervisely.geometry.mask_3d.Mask3D`], optional
    :param custom_data: Custom data associated with the :class:`~supervisely.volume_annotation.volume_object.VolumeObject`.
    :type custom_data: dict, optional

    :Usage Example:

        .. code-block:: python

            import supervisely as sly

            obj_class_heart = sly.ObjClass('heart', sly.Rectangle)
            volume_obj_heart = sly.VolumeObject(obj_class_heart)
            volume_obj_heart_json = volume_obj_heart.to_json()
            print(volume_obj_heart_json)
            # Output: {
            #     "key": "6b819f1840f84d669b32cdec225385f0",
            #     "classTitle": "heart",
            #     "tags": []
            # }
    """

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
