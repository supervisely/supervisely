# coding: utf-8

from typing import Optional

from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely.volume_annotation.constants import FIGURES, INDEX


class Slice(Frame):
    """Single slice in volume plane; holds VolumeFigures at given index. Immutable."""

    figure_type = VolumeFigure

    def __init__(self, index: int, figures: Optional[list] = None):
        """
        Same parameters as :class:`~supervisely.video_annotation.frame.Frame`.

        :param index: Slice index.
        :type index: int
        :param figures: List of VolumeFigures.
        :type figures: Optional[List[:class:`~supervisely.volume_annotation.volume_figure.VolumeFigure`]]

        :Usage Example:

            .. code-block:: python

                import supervisely as sly

                obj_class = sly.ObjClass('car', sly.Rectangle)
                volume_obj = sly.VolumeObject(obj_class)
                geometry = sly.Rectangle(0, 0, 100, 100)
                figure = sly.VolumeFigure(volume_obj, geometry, "axial", 7)
                slice_ = sly.Slice(7, figures=[figure])
        """
        super().__init__(index, figures)

    # @classmethod
    # def from_json(cls, data, objects, slices_count=None, key_id_map=None):
    #     raise NotImplementedError()
    #     _frame = super().from_json(data, objects, slices_count, key_id_map)
    #     return cls(index=_frame.index, figures=_frame.figures)

    @classmethod
    def from_json(
        cls,
        data: dict,
        objects: VolumeObjectCollection,
        plane_name: str,
        slices_count: Optional[int] = None,
        key_id_map: Optional[KeyIdMap] = None,
    ):
        """
        Deserialize a Slice object from a JSON representation.

        :param data: The JSON representation of the Slice.
        :type data: dict
        :param objects: A collection of objects in volume.
        :type objects: :class:`~supervisely.volume_annotation.volume_object_collection.VolumeObjectCollection`
        :param plane_name: The name of the plane.
        :type plane_name: str
        :param slices_count: The total number of slices in the volume, if known.
        :type slices_count: Optional[int]
        :param key_id_map: A mapping of keys to IDs used for referencing objects.
        :type key_id_map: Optional[:class:`~supervisely.video_annotation.key_id_map.KeyIdMap`]
        :returns: The deserialized Slice object.
        :rtype: :class:`~supervisely.volume_annotation.slice.Slice`
        :raises ValueError: If the slice index is negative or greater than the total number of slices.

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                slice_index = 7
                geometry = sly.Rectangle(0, 0, 100, 100)
                class_car = sly.ObjClass('car', sly.Rectangle)
                object_car = sly.VolumeObject(class_car)
                objects = sly.VolumeObjectCollection([object_car])
                figure_car = sly.VolumeFigure(object_car, geometry, sly.Plane.AXIAL, slice_index)

                slice = sly.Slice(slice_index, figures=[figure_car])
                slice_json = slice.to_json()

                new_slice = sly.Slice.from_json(slice_json, objects, sly.Plane.AXIAL)
        """

        index = data[INDEX]
        if index < 0:
            raise ValueError("Frame Index have to be >= 0")

        if slices_count is not None:
            if index > slices_count:
                raise ValueError(
                    "Item contains {} frames. Frame index is {}".format(slices_count, index)
                )

        figures = []
        for figure_json in data.get(FIGURES, []):
            figure = cls.figure_type.from_json(figure_json, objects, plane_name, index, key_id_map)
            figures.append(figure)
        return cls(index=index, figures=figures)
