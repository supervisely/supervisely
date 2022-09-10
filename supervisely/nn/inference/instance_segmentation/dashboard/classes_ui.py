from supervisely.app.widgets import Card, Text, Select, Field, Container, ObjClassView
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
        ObjClassView(ObjClass("person", AnyGeometry)),
        ObjClassView(ObjClass("person", Rectangle)),
        ObjClassView(ObjClass("person", Polygon)),
        ObjClassView(ObjClass("person", Bitmap)),
        ObjClassView(ObjClass("person", Point)),
        ObjClassView(ObjClass("person", Polyline)),
        ObjClassView(ObjClass("person", Cuboid)),
        # ObjClassView(ObjClass("person", GraphNodes, geometry_config={"1": "2"})),
        ObjClassView(ObjClass("person", Point3d)),
        ObjClassView(ObjClass("person", Cuboid3d)),
        ObjClassView(ObjClass("person", Pointcloud)),
        ObjClassView(ObjClass("person", ClosedSurfaceMesh)),
        ObjClassView(ObjClass("person", MultichannelBitmap)),
    ]
)

classes_card = Card(
    "Model classes", "Model predicts the following classes", container=model_classes
)

classes_layout = Container([classes_card])
