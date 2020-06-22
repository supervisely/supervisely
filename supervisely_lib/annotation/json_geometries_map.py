# coding: utf-8
from supervisely_lib.geometry.bitmap import Bitmap
from supervisely_lib.geometry.cuboid import Cuboid
from supervisely_lib.geometry.point import Point
from supervisely_lib.geometry.polygon import Polygon
from supervisely_lib.geometry.polyline import Polyline
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.graph import GraphNodes
from supervisely_lib.geometry.any_geometry import AnyGeometry
from supervisely_lib.geometry.cuboid_3d import Cuboid3d
from supervisely_lib.geometry.pointcloud import Pointcloud
from supervisely_lib.geometry.point_3d import Point3d
from supervisely_lib.geometry.multichannel_bitmap import MultichannelBitmap


_INPUT_GEOMETRIES = [Bitmap, Cuboid, Point, Polygon, Polyline, Rectangle, GraphNodes, AnyGeometry,
                     Cuboid3d, Pointcloud, Point3d, MultichannelBitmap]
_JSON_SHAPE_TO_GEOMETRY_TYPE = {geometry.geometry_name(): geometry for geometry in _INPUT_GEOMETRIES}


def GET_GEOMETRY_FROM_STR(figure_shape: str):
    '''
    The function create geometry class object from given string
    '''
    geometry = _JSON_SHAPE_TO_GEOMETRY_TYPE[figure_shape]
    return geometry
