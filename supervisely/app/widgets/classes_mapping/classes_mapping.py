from typing import Optional, Union, List
from supervisely.app.widgets import Widget, NotificationBox, Button, generate_id
from supervisely import ObjClass, ObjClassCollection
from supervisely.app import DataJson, StateJson

from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.alpha_mask import AlphaMask
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


type_to_shape_text = {
    AnyGeometry: "any shape",
    Rectangle: "rectangle",
    Polygon: "polygon",
    Bitmap: "bitmap (mask)",
    AlphaMask: "alpha mask",
    Polyline: "polyline",
    Point: "point",
    Cuboid: "cuboid",  #
    Cuboid3d: "cuboid 3d",
    Pointcloud: "pointcloud",  #  # "zmdi zmdi-border-clear"
    MultichannelBitmap: "n-channel mask",  # "zmdi zmdi-collection-item"
    Point3d: "point 3d",  # "zmdi zmdi-select-all"
    GraphNodes: "keypoints",
    ClosedSurfaceMesh: "volume (3d mask)",
}


class ClassesMapping(Widget):
    def __init__(
        self,
        classes: Optional[Union[List[ObjClass], ObjClassCollection]] = [],
        empty_notification: Optional[NotificationBox] = None,
        widget_id: Optional[str] = None,
    ):
        if empty_notification is None:
            empty_notification = NotificationBox(
                title="No classes",
                description="No classes to map.",
            )
        self.empty_notification = empty_notification
        self._classes = classes

        self._select_all_btn = Button()
        self._deselect_all_btn = Button()

        self._select_all_btn = Button(
            "Select all",
            button_type="text",
            show_loading=False,
            icon="zmdi zmdi-check-all",
            widget_id=generate_id(),
        )
        self._deselect_all_btn = Button(
            "Deselect all",
            button_type="text",
            show_loading=False,
            icon="zmdi zmdi-square-o",
            widget_id=generate_id(),
        )

        @self._select_all_btn.click
        def _select_all_btn_clicked():
            self.select_all()

        @self._deselect_all_btn.click
        def _deselect_all_btn_clicked():
            self.deselect_all()

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "classes": [
                {
                    **cls.to_json(),
                    "shape_text": type_to_shape_text.get(cls.geometry_type).upper(),
                    "default_value": cls.name,
                }
                for cls in self._classes
            ]
        }

    def get_json_state(self):
        return {
            "classes_values": [
                {
                    "value": cls.name,
                    "default": True,
                    "ignore": False,
                    "selected": False,
                }
                for cls in self._classes
            ]
        }

    def set(self, classes):
        self._classes = classes
        self.update_data()
        DataJson().send_changes()
        cur_mapping = self.get_mapping()
        new_mapping_values = []
        for cls in self._classes:
            value = cur_mapping.get(
                cls.name,
                {
                    "value": cls.name,
                    "default": False,
                    "ignore": True,
                    "selected": False,
                },
            )
            new_mapping_values.append(value)
        StateJson()[self.widget_id]["classes_values"] = new_mapping_values
        StateJson().send_changes()

    def get_classes(self):
        return self._classes

    def get_mapping(self):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        if len(classes_values) != len(self._classes):
            self.update_state()
            return self.get_mapping()
        mapping = {
            cls.name: classes_values[idx]
            for idx, cls in enumerate(self._classes)
            if classes_values[idx]["selected"]
        }
        return mapping

    def ignore(self, indexes: List[int]):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        for idx in indexes:
            classes_values[idx] = {"value": "", "default": False, "ignore": True}
        StateJson()[self.widget_id]["classes_values"] = classes_values
        StateJson().send_changes()

    def set_default(self):
        self.update_state()
        StateJson().send_changes()

    def set_mapping(self, mapping: dict):
        cur_mapping = self.get_mapping()
        new_mapping_values = []
        for cls in self._classes:
            cur_value = cur_mapping.get(cls.name, {"value": ""}).get("value")
            new_value = mapping.get(cls.name, cur_value)
            new_mapping_values.append(
                {
                    "value": new_value if new_value != "" else cls.name,
                    "default": new_value == cls.name,
                    "ignore": new_value == "",
                    "selected": new_value != "",
                }
            )
        StateJson()[self.widget_id]["classes_values"] = new_mapping_values
        StateJson().send_changes()

    def select_all(self):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        classes_values: dict
        for value in classes_values:
            value["selected"] = True
        StateJson()[self.widget_id]["classes_values"] = classes_values
        StateJson().send_changes()

    def deselect_all(self):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        classes_values: dict
        for value in classes_values:
            value["selected"] = False
        StateJson()[self.widget_id]["classes_values"] = classes_values
        StateJson().send_changes()

    def select(self, classes):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        classes_values: list
        for idx, cls in enumerate(self._classes):
            classes_values[idx]["selected"] = cls.name in classes
        StateJson()[self.widget_id]["classes_values"] = classes_values
        StateJson().send_changes()
