from typing import Callable, List, Optional, Union

from supervisely import ObjClass, ObjClassCollection
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import NotificationBox, Text, Widget
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.closed_surface_mesh import ClosedSurfaceMesh
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
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.color import generate_rgb

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
    Pointcloud: "pointcloud",  # "zmdi zmdi-border-clear"
    MultichannelBitmap: "n-channel mask",  # "zmdi zmdi-collection-item"
    Point3d: "point 3d",  # "zmdi zmdi-select-all"
    GraphNodes: "keypoints",
    ClosedSurfaceMesh: "volume (3d mask)",
    Mask3D: "3d mask",
    OrientedBBox: "oriented bbox",
}

shape_text_to_type = {v: k for k, v in type_to_shape_text.items()}

# Geometry types available for creating new classes (excluding GraphNodes)
available_geometry_types = [
    {"value": "rectangle", "label": "Rectangle"},
    {"value": "polygon", "label": "Polygon"},
    {"value": "bitmap (mask)", "label": "Bitmap (mask)"},
    {"value": "polyline", "label": "Polyline"},
    {"value": "point", "label": "Point"},
    {"value": "any shape", "label": "Any shape"},
    {"value": "oriented bbox", "label": "Oriented Bounding Box"},
]


class ClassesListSelector(Widget):
    class Routes:
        CHECKBOX_CHANGED = "checkbox_cb"
        CLASS_CREATED = "class_created_cb"

    def __init__(
        self,
        classes: Optional[Union[List[ObjClass], ObjClassCollection]] = [],
        multiple: Optional[bool] = False,
        empty_notification: Optional[NotificationBox] = None,
        allow_new_classes: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        # Convert to list for internal use to allow mutations when adding new classes
        if isinstance(classes, ObjClassCollection):
            self._classes = list(classes)
        else:
            self._classes = list(classes) if classes else []
        self._multiple = multiple
        self._allow_new_classes = allow_new_classes
        self._class_created_handled = False
        if empty_notification is None:
            empty_notification = NotificationBox(
                title="No classes",
                description="No classes to select.",
            )
        self.empty_notification = empty_notification
        self._error_message = Text("", status="error", font_size=13)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        classes_list = []
        for cls in self._classes:
            shape_text = type_to_shape_text.get(cls.geometry_type)
            class_dict = {**cls.to_json(), "shape_text": shape_text.upper() if shape_text else ""}
            classes_list.append(class_dict)
        return {
            "classes": classes_list,
            "availableGeometryTypes": available_geometry_types,
        }

    def get_json_state(self):
        return {
            "selected": [False for _ in self._classes],
            "createClassDialog": {
                "visible": False,
                "className": "",
                "geometryType": "rectangle",
                "showError": False,
            },
        }

    def set(self, classes: Union[List[ObjClass], ObjClassCollection]):
        selected_classes = [cls.name for cls in self.get_selected_classes()]
        # Convert to list for internal use
        if isinstance(classes, ObjClassCollection):
            self._classes = list(classes)
        else:
            self._classes = list(classes) if classes else []
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

    def _show_error(self, message: str):
        """Show error message in the create class dialog."""
        self._error_message.text = message
        StateJson()[self.widget_id]["createClassDialog"]["showError"] = True
        StateJson().send_changes()

    def _hide_dialog(self):
        """Hide the create class dialog and reset its state."""
        state_obj = StateJson()[self.widget_id]["createClassDialog"]
        state_obj["visible"] = False
        state_obj["className"] = ""
        state_obj["showError"] = False
        StateJson().send_changes()

    def _add_new_class(self, new_class: ObjClass):
        """Add a new class to the widget and update the UI."""
        # Add to classes list
        self._classes.append(new_class)

        # Add selection state for the new class (selected by default)
        StateJson()[self.widget_id]["selected"].append(True)

        # Update data to reflect the new class in the UI
        self.update_data()
        DataJson().send_changes()
        StateJson().send_changes()

    def selection_changed(self, func):
        route_path = self.get_route_path(ClassesListSelector.Routes.CHECKBOX_CHANGED)
        server = self._sly_app.get_server()
        self._checkboxes_handled = True

        @server.post(route_path)
        def _click():
            selected = self.get_selected_classes()
            func(selected)

        return _click

    def class_created(self, func: Callable[[ObjClass], None]):
        """
        Decorator to handle new class creation event.
        The decorated function receives the newly created ObjClass.

        :param func: Function to be called when a new class is created
        :type func: Callable[[ObjClass], None]
        """
        route_path = self.get_route_path(ClassesListSelector.Routes.CLASS_CREATED)
        server = self._sly_app.get_server()
        self._class_created_handled = True

        @server.post(route_path)
        def _class_created():
            state = StateJson()[self.widget_id]["createClassDialog"]
            class_name = state["className"].strip()
            geometry_type_str = state["geometryType"]

            if not class_name:
                self._show_error("Class name cannot be empty")
                return

            if any(cls.name == class_name for cls in self._classes):
                self._show_error(f"Class '{class_name}' already exists")
                return

            geometry_type = shape_text_to_type.get(geometry_type_str)
            if geometry_type is None:
                self._show_error("Invalid geometry type")
                return

            # Generate color for the new class
            existing_colors = [cls.color for cls in self._classes]
            new_color = generate_rgb(existing_colors)

            # Create new class
            new_class = ObjClass(name=class_name, geometry_type=geometry_type, color=new_color)

            self._add_new_class(new_class)
            self._hide_dialog()

            func(new_class)

        return _class_created
