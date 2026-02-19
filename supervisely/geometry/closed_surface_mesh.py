from supervisely.geometry.geometry import Geometry
from supervisely.geometry.constants import (
    EXTERIOR,
    INTERIOR,
    POINTS,
    LABELER_LOGIN,
    UPDATED_AT,
    CREATED_AT,
    ID,
    CLASS_ID,
)


class ClosedSurfaceMesh(Geometry):
    """3D closed surface mesh (triangular faces). Immutable."""

    @staticmethod
    def geometry_name():
        """
        Returns the name of the geometry.

        :returns: name of the geometry
        :rtype: str
        """
        return "closed_surface_mesh"

    def draw(self, bitmap, color, thickness=1, config=None):
        """
        Draws the ClosedSurfaceMesh on a bitmap.

        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        :type config: dict
        """
        raise NotImplementedError('Method "draw" is unavailable for this geometry')

    def draw_contour(self, bitmap, color, thickness=1, config=None):
        """
        Draws the contour of the ClosedSurfaceMesh on a bitmap.

        :param bitmap: np.ndarray
        :param color: [R, G, B]
        :param thickness: (int)
        :param config: drawing config specific to a concrete subclass, e.g. per edge colors
        :type config: dict
        """
        raise NotImplementedError(
            'Method "draw_contour" is unavailable for this geometry'
        )

    def convert(self, new_geometry, contour_radius=0, approx_epsilon=None):
        """
        Converts the ClosedSurfaceMesh to a new geometry.

        :param new_geometry: new geometry
        :type new_geometry: :class:`~supervisely.geometry.geometry.Geometry`
        :param contour_radius: radius of the contour
        :type contour_radius: int
        :param approx_epsilon: epsilon for the approximation
        :type approx_epsilon: float
        """
        raise NotImplementedError('Method "convert" is unavailable for this geometry')

    def to_json(self):
        """
        Converts the ClosedSurfaceMesh to a JSON object.

        :returns: JSON object
        :rtype: dict
        """
        res = {}
        self._add_creation_info(res)
        return res

    @classmethod
    def from_json(cls, data):
        """
        Converts a JSON object to a ClosedSurfaceMesh.

        :param data: JSON object
        :type data: dict
        :returns: ClosedSurfaceMesh
        :rtype: :class:`~supervisely.geometry.closed_surface_mesh.ClosedSurfaceMesh`
        """
        labeler_login = data.get(LABELER_LOGIN, None)
        updated_at = data.get(UPDATED_AT, None)
        created_at = data.get(CREATED_AT, None)
        sly_id = data.get(ID, None)
        class_id = data.get(CLASS_ID, None)

        return cls(
            sly_id=sly_id,
            class_id=class_id,
            labeler_login=labeler_login,
            updated_at=updated_at,
            created_at=created_at,
        )
