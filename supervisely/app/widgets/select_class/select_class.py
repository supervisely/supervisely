from typing import Callable, List, Optional, Union

from supervisely import ObjClass, ObjClassCollection
from supervisely.app import DataJson, StateJson
from supervisely.app.widgets import Text, Widget
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

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

type_to_shape_text = {
    AnyGeometry: "any shape",
    Rectangle: "rectangle",
    Polygon: "polygon",
    AlphaMask: "alpha mask",
    Bitmap: "bitmap (mask)",
    Polyline: "polyline",
    Point: "point",
    Cuboid2d: "cuboid 2d",
    Cuboid3d: "cuboid 3d",
    Pointcloud: "pointcloud",
    MultichannelBitmap: "n-channel mask",
    Point3d: "point 3d",
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


class SelectClass(Widget):
    """
    SelectClass is a compact dropdown widget for selecting object classes with an option to create
    new classes on the fly.

    :param classes: Initial list of ObjClass instances
    :type classes: Optional[Union[List[ObjClass], ObjClassCollection]]
    :param filterable: Enable search/filter functionality in dropdown
    :type filterable: Optional[bool]
    :param placeholder: Placeholder text when no class is selected
    :type placeholder: Optional[str]
    :param show_add_new_class: Show "Add new class" option at the end of the list
    :type show_add_new_class: Optional[bool]
    :param size: Size of the select dropdown
    :type size: Optional[Literal["large", "small", "mini"]]
    :param multiple: Enable multiple selection
    :type multiple: bool
    :param widget_id: Unique widget identifier
    :type widget_id: Optional[str]

    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely.app.widgets import SelectClass

        # Create some initial classes
        class_car = sly.ObjClass('car', sly.Rectangle, color=[255, 0, 0])
        class_person = sly.ObjClass('person', sly.Polygon, color=[0, 255, 0])

        # Create SelectClass widget
        select_class = SelectClass(
            classes=[class_car, class_person],
            filterable=True,
            show_add_new_class=True
        )

        # Handle selection changes
        @select_class.value_changed
        def on_class_selected(class_name):
            print(f"Selected class: {class_name}")
            selected_class = select_class.get_selected_class()
            print(f"Class object: {selected_class}")

        # Handle new class creation
        @select_class.class_created
        def on_class_created(new_class: sly.ObjClass):
            print(f"New class created: {new_class.name}")
            # Optionally update your project meta or perform other actions
    """

    class Routes:
        VALUE_CHANGED = "value_changed"
        CLASS_CREATED = "class_created_cb"

    def __init__(
        self,
        classes: Optional[Union[List[ObjClass], ObjClassCollection]] = [],
        filterable: Optional[bool] = True,
        placeholder: Optional[str] = "Select class",
        show_add_new_class: Optional[bool] = True,
        size: Optional[Literal["large", "small", "mini"]] = None,
        multiple: bool = False,
        widget_id: Optional[str] = None,
    ):
        # Convert to list for internal use to allow mutations when adding new classes
        if isinstance(classes, ObjClassCollection):
            self._classes = list(classes)
        else:
            self._classes = list(classes) if classes else []

        self._filterable = filterable
        self._placeholder = placeholder
        self._show_add_new_class = show_add_new_class
        self._size = size
        self._multiple = multiple

        self._changes_handled = False
        self._class_created_callback = None

        # Store error message widget
        self._error_message = Text("", status="error", font_size=13)

        # Initialize parent Widget
        super().__init__(widget_id=widget_id, file_path=__file__)

        # Register class_created route if show_add_new_class is enabled
        if self._show_add_new_class:
            self._register_class_created_route()

    def get_json_data(self):
        """Build JSON data for the widget."""
        # Build items list with class info
        items = []
        for cls in self._classes:
            shape_text = type_to_shape_text.get(cls.geometry_type, "")
            items.append(
                {
                    "value": cls.name,
                    "label": cls.name,
                    "color": cls.color,
                    "geometryType": shape_text.upper() if shape_text else "",
                }
            )

        return {
            "items": items,
            "placeholder": self._placeholder,
            "filterable": self._filterable,
            "multiple": self._multiple,
            "size": self._size,
            "showAddNewClass": self._show_add_new_class,
            "availableGeometryTypes": available_geometry_types,
        }

    def get_json_state(self):
        """Build JSON state for the widget."""
        return {
            "value": self._classes[0].name if self._classes else None,
            "createClassDialog": {
                "visible": False,
                "className": "",
                "geometryType": "rectangle",
                "showError": False,
            },
        }

    def get_value(self) -> Union[str, List[str], None]:
        """Get the currently selected class name(s)."""
        return StateJson()[self.widget_id]["value"]

    def get_selected_class(self) -> Union[ObjClass, List[ObjClass], None]:
        """Get the currently selected ObjClass object(s)."""
        value = self.get_value()
        if value is None:
            return None

        if self._multiple:
            if not isinstance(value, list):
                return []
            result = []
            for class_name in value:
                for cls in self._classes:
                    if cls.name == class_name:
                        result.append(cls)
                        break
            return result
        else:
            for cls in self._classes:
                if cls.name == value:
                    return cls
            return None

    def set_value(self, class_name: Union[str, List[str]]):
        """Set the selected class by name."""
        StateJson()[self.widget_id]["value"] = class_name
        StateJson().send_changes()

    def get_all_classes(self) -> List[ObjClass]:
        """Get all available classes."""
        return self._classes.copy()

    def set(self, classes: Union[List[ObjClass], ObjClassCollection]):
        """Update the list of available classes."""
        # Convert to list for internal use
        if isinstance(classes, ObjClassCollection):
            self._classes = list(classes)
        else:
            self._classes = list(classes) if classes else []

        # Update data
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

        # Reset value if current selection is not in new classes
        current_value = StateJson()[self.widget_id]["value"]
        if current_value:
            if self._multiple:
                if isinstance(current_value, list):
                    # Keep only valid selections
                    valid = [
                        v for v in current_value if any(cls.name == v for cls in self._classes)
                    ]
                    if valid != current_value:
                        StateJson()[self.widget_id]["value"] = valid
                        StateJson().send_changes()
            else:
                if not any(cls.name == current_value for cls in self._classes):
                    StateJson()[self.widget_id]["value"] = (
                        self._classes[0].name if self._classes else None
                    )
                    StateJson().send_changes()

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

        # Update data
        DataJson()[self.widget_id] = self.get_json_data()
        DataJson().send_changes()

        # Set the new class as selected
        if self._multiple:
            current = StateJson()[self.widget_id]["value"]
            if isinstance(current, list):
                current.append(new_class.name)
            else:
                StateJson()[self.widget_id]["value"] = [new_class.name]
        else:
            StateJson()[self.widget_id]["value"] = new_class.name
        StateJson().send_changes()

    def value_changed(self, func: Callable[[Union[ObjClass, List[ObjClass]]], None]):
        """
        Decorator to handle value change event.
        The decorated function receives the selected ObjClass (or list of ObjClass if multiple=True).

        :param func: Function to be called when selection changes
        :type func: Callable[[Union[ObjClass, List[ObjClass]]], None]
        """
        route_path = self.get_route_path(SelectClass.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            selected = self.get_selected_class()
            if selected is not None:
                func(selected)

        return _value_changed

    def _register_class_created_route(self):
        """Register the class_created route."""
        route_path = self.get_route_path(SelectClass.Routes.CLASS_CREATED)
        server = self._sly_app.get_server()

        @server.post(route_path)
        def _class_created():
            state = StateJson()[self.widget_id]["createClassDialog"]
            class_name = state["className"].strip()
            geometry_type_str = state["geometryType"]

            # Validate class name
            if not class_name:
                self._show_error("Class name cannot be empty")
                return

            # Check if class with this name already exists
            if any(cls.name == class_name for cls in self._classes):
                self._show_error(f"Class '{class_name}' already exists")
                return

            # Get geometry type from string
            geometry_type = shape_text_to_type.get(geometry_type_str)
            if geometry_type is None:
                self._show_error("Invalid geometry type")
                return

            # Generate color for the new class
            existing_colors = [cls.color for cls in self._classes]
            new_color = generate_rgb(existing_colors)

            # Create new class
            new_class = ObjClass(name=class_name, geometry_type=geometry_type, color=new_color)

            # Add to widget
            self._add_new_class(new_class)

            # Hide dialog
            self._hide_dialog()

            # Call user's callback if set
            if self._class_created_callback:
                self._class_created_callback(new_class)

    def class_created(self, func: Callable[[ObjClass], None]):
        """
        Decorator to handle new class creation event.
        The decorated function receives the newly created ObjClass.

        :param func: Function to be called when a new class is created
        :type func: Callable[[ObjClass], None]
        """
        self._class_created_callback = func
        return func
