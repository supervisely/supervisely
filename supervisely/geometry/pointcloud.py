# coding: utf-8

from supervisely.geometry.constants import LABELER_LOGIN, UPDATED_AT, CREATED_AT, ID, CLASS_ID, INDICES
from supervisely.geometry.geometry import Geometry

# pointcloud mask (segmentation)
class Pointcloud(Geometry):
    """3D point cloud geometry: reference to external point cloud file. Immutable."""

    @staticmethod
    def geometry_name():
        """
        Returns the name of the geometry.

        :returns: name of the geometry
        :rtype: str
        """
        return 'point_cloud'

    def __init__(self, indices, sly_id=None, class_id=None, labeler_login=None, updated_at=None, created_at=None):
        """Pointcloud initialization.

        :param indices: Indices of the pointcloud.
        :type indices: list
        :param sly_id: Pointcloud ID in Supervisely server.
        :type sly_id: int, optional
        :param class_id: ID of ObjClass to which Pointcloud belongs.
        :type class_id: int, optional
        :param labeler_login: Login of the user who created Pointcloud.
        :type labeler_login: str, optional
        :param updated_at: Date and Time when Pointcloud was modified last. Date Format: Year:Month:Day:Hour:Minute:Seconds. Example: '2021-01-22T19:37:50.158Z'.
        :type updated_at: str, optional
        :param created_at: Date and Time when Pointcloud was created. Date Format is the same as in "updated_at" parameter.
        :type created_at: str, optional
        """
        super().__init__(sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)

        if type(indices) is not list:
            raise TypeError("\"indices\" param has to be of type {!r}".format(type(list)))

        self._indices = indices

    @property
    def indices(self):
        """
        Copy of the indices of the Pointcloud.

        :returns: indices of the :class:`~supervisely.geometry.pointcloud.Pointcloud`
        :rtype: list
        """
        return self._indices.copy()

    def to_json(self):
        """
        Converts the Pointcloud to a JSON object.

        :returns: JSON object
        :rtype: dict
        :returns: Pointcloud in json format
        """
        res = {INDICES: self.indices}
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        Converts a JSON object to a Pointcloud.

        :param data: JSON object
        :type data: dict
        :returns: Pointcloud
        :rtype: :class:`~supervisely.geometry.pointcloud.Pointcloud`
        :returns: Pointcloud from json.
        :rtype: :class:`~supervisely.geometry.pointcloud.Pointcloud`
        """
        indices = data[INDICES]

        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)
        return cls(indices, sly_id=sly_id, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
