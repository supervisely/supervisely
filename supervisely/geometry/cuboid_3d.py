# coding: utf-8

from typing import Optional
from copy import deepcopy

from supervisely.geometry.constants import X, Y, Z, \
    POSITION, ROTATION, DIMENTIONS, LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID
from supervisely.geometry.geometry import Geometry


class Vector3d:
    """
    A simple 3D vector (x, y, z) used by :class:`~supervisely.geometry.cuboid_3d.Cuboid3d`.

    Stores three coordinates and supports JSON (de)serialization via :meth:`to_json` / :meth:`from_json`.
    """
    def __init__(self, x, y, z):
        """
        :param x: int
        :param y: int
        :param z: int
        """
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        """
        """
        return self._x

    @property
    def y(self):
        """
        """
        return self._y

    @property
    def z(self):
        """
        """
        return self._z

    def to_json(self):
        """
        The function to_json convert Vector3d class object to json format(dict)
        :returns: Vector3d in json format
        """
        return {X: self.x, Y: self.y, Z: self.z}

    @classmethod
    def from_json(cls, data):
        """
        The function from_json convert Vector3d from json format(dict) to Vector3d class object.
        :param data: Vector3d in json format(dict)
        :returns: Vector3d from json.
        :rtype: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        """
        x = data[X]
        y = data[Y]
        z = data[Z]
        return cls(x, y, z)

    def clone(self):
        """
        """
        return deepcopy(self)


class Cuboid3d(Geometry):
    """3D cuboid with position, rotation, and dimensions (Vector3d). Immutable."""

    @staticmethod
    def geometry_name():
        """
        Returns the name of the geometry.

        :returns: name of the geometry
        :rtype: str
        """
        return 'cuboid_3d'

    def __init__(
        self,
        position: Vector3d,
        rotation: Vector3d,
        dimensions: Vector3d,
        sly_id: Optional[int] = None,
        class_id: Optional[int] = None,
        labeler_login: Optional[int] = None,
        updated_at: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        """
        Cuboid3d is a geometry for a single :class:`~supervisely.annotation.label.Label`. :class:`~supervisely.geometry.cuboid_3d.Cuboid3d` object is immutable.

        :param position: Vector3d.
        :type position: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        :param rotation: Vector3d.
        :type rotation: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        :param dimensions: Vector3d.
        :type dimensions: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        :param sly_id: Cuboid3d ID in Supervisely server.
        :type sly_id: int, optional
        :param class_id: ID of ObjClass to which Cuboid3d belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created Cuboid3d.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when Cuboid3d was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when Cuboid3d was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        """
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

        if type(position) is not Vector3d:
            raise TypeError("\"position\" param has to be of type {!r}".format(type(Vector3d)))
        if type(rotation) is not Vector3d:
            raise TypeError("\"rotation\" param has to be of type {!r}".format(type(Vector3d)))
        if type(dimensions) is not Vector3d:
            raise TypeError("\"dimensions\" param has to be of type {!r}".format(type(Vector3d)))

        self._position = position
        self._rotation = rotation
        self._dimensions = dimensions

    @property
    def position(self):
        """
        Copy of the position of the Cuboid3d.

        :returns: Position of the :class:`~supervisely.geometry.cuboid_3d.Cuboid3d`
        :rtype: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        """
        return self._position.clone()

    @property
    def rotation(self):
        """
        Copy of the rotation of the Cuboid3d.

        :returns: Rotation of the :class:`~supervisely.geometry.cuboid_3d.Cuboid3d`
        :rtype: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        """
        return self._rotation.clone()

    @property
    def dimensions(self):
        """
        Copy of the dimensions of the Cuboid3d.

        :returns: Dimensions of the :class:`~supervisely.geometry.cuboid_3d.Cuboid3d`
        :rtype: :class:`~supervisely.geometry.cuboid_3d.Vector3d`
        """
        return self._dimensions.clone()

    def to_json(self):
        """
        Converts the Cuboid3d to a JSON object.

        :returns: JSON object
        :rtype: dict
        :returns: Cuboid3d in json format
        """
        res = {POSITION: self.position.to_json(),
                ROTATION: self.rotation.to_json(),
                DIMENTIONS: self.dimensions.to_json()}

        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        Converts a JSON object to a Cuboid3d.

        :param data: JSON object
        :type data: dict
        :returns: Cuboid3d
        :rtype: :class:`~supervisely.geometry.cuboid_3d.Cuboid3d`
        :param data: Cuboid3d in json format(dict)
        :returns: Cuboid3d from json.
        :rtype: :class:`~supervisely.geometry.cuboid_3d.Cuboid3d`
        """
        position = Vector3d.from_json(data[POSITION])
        rotation = Vector3d.from_json(data[ROTATION])
        dimentions = Vector3d.from_json(data[DIMENTIONS])

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(position, rotation, dimentions, sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
