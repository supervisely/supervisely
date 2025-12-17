# coding: utf-8
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.cuboid_2d import Cuboid2d
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.mask_3d import Mask3D
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.oriented_bbox import OrientedBBox
from supervisely.geometry.point import Point
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.polyline_3d import Polyline3D
from supervisely.geometry.rectangle import Rectangle

_INPUT_GEOMETRIES = [
    Bitmap,
    Mask3D,
    Cuboid,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    GraphNodes,
    AnyGeometry,
    Cuboid3d,
    Pointcloud,
    Point3d,
    MultichannelBitmap,
    ClosedSurfaceMesh,
    AlphaMask,
    Cuboid2d,
    Polyline3D,
    OrientedBBox,
]
_JSON_SHAPE_TO_GEOMETRY_TYPE = {
    geometry.geometry_name(): geometry for geometry in _INPUT_GEOMETRIES
}


def GET_GEOMETRY_FROM_STR(figure_shape: str):
    """
    The function create geometry class object from given string
    """
    if figure_shape not in _JSON_SHAPE_TO_GEOMETRY_TYPE.keys():
        raise KeyError(
            f"Unknown shape: '{figure_shape}'. Supported shapes: {list(_JSON_SHAPE_TO_GEOMETRY_TYPE.keys())}"
        )
    geometry = _JSON_SHAPE_TO_GEOMETRY_TYPE[figure_shape]
    return geometry
