# coding: utf-8

from supervisely.geometry.cuboid_3d import Vector3d
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID


class Point3d(Geometry):
    """
    Point3d is a geometry for a single :class:`~supervisely.annotation.label.Label`. :class:`~supervisely.geometry.point_3d.Point3d` object is immutable.
    """

    @staticmethod
    def geometry_name():
        """
        Returns the name of the geometry.

        :returns: name of the geometry
        :rtype: str
        """
        return 'point_3d'

    def __init__(self, point: Vector3d,
                 sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        """
        Point3d is a geometry for a single :class:`~supervisely.annotation.label.Label`. :class:`~supervisely.geometry.point_3d.Point3d` object is immutable.

        :param point: Vector3d.
        :type point: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        :param sly_id: Point3d ID in Supervisely server.
        :type sly_id: int, optional
        :param class_id: ID of ObjClass to which Point3d belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created Point3d.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when Point3d was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when Point3d was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        """
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

        if type(point) is not Vector3d:
            raise TypeError("\"position\" param has to be of type {!r}".format(type(Vector3d)))

        self._point = point

    @property
    def x(self):
        """
        Copy of the x coordinate of the Point3d.

        :returns: x coordinate of the :class:`~supervisely.geometry.point_3d.Point3d`
        :rtype: int
        """
        return self._point._x

    @property
    def y(self):
        """
        Copy of the y coordinate of the Point3d.

        :returns: y coordinate of the :class:`~supervisely.geometry.point_3d.Point3d`
        :rtype: int
        """
        return self._point._y

    @property
    def z(self):
        """
        Copy of the z coordinate of the Point3d.

        :returns: z coordinate of the :class:`~supervisely.geometry.point_3d.Point3d`
        :rtype: int
        """
        return self._point._z

    def to_json(self):
        """
        Converts the Point3d to a JSON object.

        :returns: JSON object
        :rtype: dict
        :returns: Point3d in json format
        """
        res = self._point.to_json()
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        Converts a JSON object to a Point3d.

        :param data: JSON object
        :type data: dict
        :returns: Point3d
        :rtype: :class:`~supervisely.geometry.point_3d.Point3d`
        :returns: Point3d from json.
        :rtype: :class:`~supervisely.geometry.point_3d.Point3d`
        """
        point = Vector3d.from_json(data)

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(point, sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
