from supervisely.geometry.geometry import Geometry


class ClosedSurfaceMesh(Geometry):
    @staticmethod
    def geometry_name():
        return "closed_surface_mesh"

    def draw(self, bitmap, color, thickness=1, config=None):
        raise NotImplementedError('Method "draw" is unavailable for this geometry')

    def draw_contour(self, bitmap, color, thickness=1, config=None):
        raise NotImplementedError(
            'Method "draw_contour" is unavailable for this geometry'
        )

    def convert(self, new_geometry, contour_radius=0, approx_epsilon=None):
        raise NotImplementedError('Method "convert" is unavailable for this geometry')
