from typing import List, Optional, Union

from supervisely import ObjClass, ObjClassCollection
from supervisely.app import StateJson
from supervisely.app.widgets import Button, NotificationBox, Widget, generate_id
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
from supervisely.geometry.cuboid_2d import Cuboid2d
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.mask_3d import Mask3D
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.point import Point
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle

type_to_shape_text = {
    AnyGeometry: "any shape",
    Rectangle: "rectangle",
    Polygon: "polygon",
    AlphaMask: "alpha mask",
    Bitmap: "bitmap (mask)",
    Polyline: "polyline",
    Point: "point",
    Cuboid2d: "cuboid 2d",  #
    Cuboid3d: "cuboid 3d",
    Pointcloud: "pointcloud",  #  # "zmdi zmdi-border-clear"
    MultichannelBitmap: "n-channel mask",  # "zmdi zmdi-collection-item"
    Point3d: "point 3d",  # "zmdi zmdi-select-all"
    GraphNodes: "keypoints",
    ClosedSurfaceMesh: "volume (3d mask)",
    Mask3D: "3d mask",
}


class ClassesListSelector(Widget):
    class Routes:
        CHECKBOX_CHANGED = "checkbox_cb"

    def __init__(
        self,
        classes: Optional[Union[List[ObjClass], ObjClassCollection]] = [],
        multiple: Optional[bool] = False,
        empty_notification: Optional[NotificationBox] = None,
        widget_id: Optional[str] = None,
    ):
        self._classes = classes
        self._multiple = multiple
        if empty_notification is None:
            empty_notification = NotificationBox(
                title="No classes",
                description="No classes to select.",
            )
        self.empty_notification = empty_notification
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        classes_list = []
        for cls in self._classes:
            shape_text = type_to_shape_text.get(cls.geometry_type)
            class_dict = {**cls.to_json(), "shape_text": shape_text.upper() if shape_text else ""}
            classes_list.append(class_dict)
        return {"classes": classes_list}

    def get_json_state(self):
        return {"selected": [False for _ in self._classes]}

    def set(self, classes: Union[List[ObjClass], ObjClassCollection]):
        selected_classes = [cls.name for cls in self.get_selected_classes()]
        self._classes = classes
        StateJson()[self.widget_id]["selected"] = [
            cls.name in selected_classes for cls in self._classes
        ]
        self.update_data()
        StateJson().send_changes()

    def get_selected_classes(self):
        selected = StateJson()[self.widget_id]["selected"]
        return [cls for cls, is_selected in zip(self._classes, selected) if is_selected]

    def select_all(self):
        StateJson()[self.widget_id]["selected"] = [True for _ in self._classes]
        StateJson().send_changes()

    def deselect_all(self):
        StateJson()[self.widget_id]["selected"] = [False for _ in self._classes]
        StateJson().send_changes()

    def select(self, names: List[str]):
        selected = [cls.name in names for cls in self._classes]
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def deselect(self, names: List[str]):
        selected = StateJson()[self.widget_id]["selected"]
        for idx, cls in enumerate(self._classes):
            if cls.name in names:
                selected[idx] = False
        StateJson()[self.widget_id]["selected"] = selected
        StateJson().send_changes()

    def set_multiple(self, value: bool):
        self._multiple = value

    def get_all_classes(self):
        return self._classes

    def selection_changed(self, func):
        route_path = self.get_route_path(ClassesListSelector.Routes.CHECKBOX_CHANGED)
        server = self._sly_app.get_server()
        self._checkboxes_handled = True

        @server.post(route_path)
        def _click():
            selected = self.get_selected_classes()
            func(selected)

        return _click
