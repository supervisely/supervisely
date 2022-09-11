from supervisely.app.widgets import (
    Card,
    Text,
    Select,
    Field,
    Container,
    ObjClassView,
    Checkbox,
)
from supervisely.annotation.obj_class import ObjClass


from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.point import Point
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh

model_classes = Container(
    [
        Checkbox(ObjClassView(ObjClass("person AnyGeometry", AnyGeometry))),
        Checkbox(ObjClassView(ObjClass("person Rectangle", Rectangle))),
        Checkbox(ObjClassView(ObjClass("person Polygon", Polygon))),
        Checkbox(ObjClassView(ObjClass("person Bitmap", Bitmap))),
        Checkbox(ObjClassView(ObjClass("person Point", Point))),
        Checkbox(ObjClassView(ObjClass("person Polyline", Polyline))),
        Checkbox(ObjClassView(ObjClass("person Cuboid", Cuboid))),
        # ObjClassView(ObjClass("person", GraphNodes, geometry_config={"1": "2"})),
        Checkbox(ObjClassView(ObjClass("person Point3d", Point3d))),
        Checkbox(ObjClassView(ObjClass("person Cuboid3d", Cuboid3d))),
        Checkbox(ObjClassView(ObjClass("person Pointcloud", Pointcloud))),
        Checkbox(ObjClassView(ObjClass("person ClosedSurfaceMesh", ClosedSurfaceMesh))),
        Checkbox(
            ObjClassView(ObjClass("person MultichannelBitmap", MultichannelBitmap))
        ),
        Checkbox(ObjClassView(ObjClass("person Cuboid3d", Cuboid3d))),
        Checkbox(ObjClassView(ObjClass("person Pointcloud", Pointcloud))),
        Checkbox(ObjClassView(ObjClass("person ClosedSurfaceMesh", ClosedSurfaceMesh))),
        Checkbox(
            ObjClassView(ObjClass("person MultichannelBitmap", MultichannelBitmap))
        ),
    ],
    direction="horizontal",
    # overflow="scroll"
    overflow="wrap",
    # grid_cell_width="25%",
)

classes_card = Card(
    "Model classes",
    "Model predicts the following classes",
    content=model_classes,
)

classes_layout = Container(
    [
        classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
        # classes_card,
    ],
    direction="horizontal",
    overflow="scroll",
    # grid_cell_width="25%",
)
