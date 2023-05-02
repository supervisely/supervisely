# coding: utf-8

# docs
from typing import Optional
from supervisely.geometry.geometry import Geometry
import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_


class Bitmap3d(Geometry):
    """
    Bitmap 3D geometry for a single :class:`Label<supervisely.annotation.label.Label>`. :class:`Bitmap3d<Bitmap3d>` object is immutable.

    :param data: Bitmap 3D mask data. Must be a numpy array with only 2 unique values: [0, 1] or [0, 255] or [False, True].
    :type data: np.ndarray
    :param sly_id: Bitmap 3D ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Bitmap 3D belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Bitmap 3D.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Bitmap 3D was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Bitmap 3D was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional
    :raises: :class:`ValueError`, if data is not bool or no pixels set to True in data
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Create simple Bitmap 3D
        bitmap3d = np.zeros((3, 3, 3), dtype=np.bool_)
        bitmap3d[0:2, 0:2, 0:2] = True

        shape = sly.Bitmap3d(bitmap3d)

        print(shape.data)
        # Output:
        #    [[[ True  True False]
        #      [ True  True False]
        #      [False False False]]

        #     [[ True  True False]
        #      [ True  True False]
        #      [False False False]]

        #     [[False False False]
        #      [False False False]
        #      [False False False]]]
    """

    def __init__(
        self,
        data: np.ndarray,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[str] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        if not isinstance(data, np.ndarray):
            raise TypeError('Bitmap 3D "data" argument must be numpy array object!')

        data_dims = len(data.shape)
        if data_dims != 3:
            raise ValueError(
                f'Bitmap 3D "data" argument must be a 3-dimensional numpy array. Instead got {data_dims} dimensions'
            )

        if data.dtype != np.bool:
            if list(np.unique(data)) not in [[0, 1], [0, 255]]:
                raise ValueError(
                    f"Bitmap 3D mask data values must be one of: [0 1], [0 255], [False True]. Instead got {np.unique(data)}."
                )

            if list(np.unique(data)) == [0, 1]:
                data = np.array(data, dtype=bool)
            elif list(np.unique(data)) == [0, 255]:
                data = np.array(data / 255, dtype=bool)

        self.data = data

    @staticmethod
    def geometry_name():
        """geometry_name"""
        return "bitmap_3d"
