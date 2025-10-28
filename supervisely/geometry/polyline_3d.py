from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely.geometry.cuboid_3d import Vector3d
from typing import List, Union


class Polyline3D(Geometry):
    """
    Polyline3D geometry

    :param points: List of 3D point coordinates which define the polyline in 3D space.
    :type points: List[List[int, int, int]]
    :param sly_id: Polyline ID in Supervisely server.
    :type sly_id: int, optional
    :param class_id: ID of :class:`ObjClass<supervisely.annotation.obj_class.ObjClass>` to which Polyline belongs.
    :type class_id: int, optional
    :param labeler_login: Login of the user who created Polyline.
    :type labeler_login: str, optional
    :param updated_at: Date and Time when Polyline was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
    :type updated_at: str, optional
    :param created_at: Date and Time when Polyline was created. Date Format is the same as in "updated_at" parameter.
    :type created_at: str, optional

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        figure = sly.Polyline(points)
    """

    @staticmethod
    def geometry_name():
        return "polyline_3d"

    def __init__(
        self,
        points: Union[List[float], List[Vector3d]],
        sly_id=None,
        class_id=None,
        labeler_login=None,
        updated_at=None,
        created_at=None,
    ):
        if not isinstance(points[0], Vector3d):
            points = [Vector3d(point[0], point[1], point[2]) for point in points]
        super().__init__(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )

        self._points = points

    @property
    def points(self):
        return self._points

    def to_json(self):
        points = [[point.x, point.y, point.z] for point in self._points]
        res = {"points": points}
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        Convert a json dict to Polyline3D.

        :param data: Polyline3D in json format as a dict.
        :type data: dict
        :return: Polyline3D object
        :rtype: :class:`Polyline3D<Polyline3D>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            figure_json = {
                "points": {
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                }
            }
            figure = sly.Polyline3D.from_json(figure_json)
        """
        if not data.get("points"):
            raise ValueError("Data dict must contain 'points' field!")
        points = data["points"]
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(
            points,
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
