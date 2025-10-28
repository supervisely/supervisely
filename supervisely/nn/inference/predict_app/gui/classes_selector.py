from typing import Any, Dict

import pandas as pd

from supervisely.app.widgets import (
    Button,
    Card,
    ClassesTable,
    Container,
    FastTable,
    Text,
)
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.point import Point
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.project.project_meta import ProjectMeta

type_to_zmdi_icon = {
    AnyGeometry: "zmdi zmdi-shape",
    Rectangle: "zmdi zmdi-crop-din",  # "zmdi zmdi-square-o"
    # sly.Polygon: "icons8-polygon",  # "zmdi zmdi-edit"
    Polygon: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAB6klEQVRYhe2Wuy8EURTGf+u5VESNXq2yhYZCoeBv8RcI1i6NVUpsoVCKkHjUGlFTiYb1mFmh2MiKjVXMudmb3cPOzB0VXzKZm5k53/nmvO6Ff4RHD5AD7gFP1l3Kd11AHvCBEpAVW2esAvWmK6t8l1O+W0lCQEnIJoAZxUnzNQNkZF36jrQjgoA+uaciCgc9VaExBOyh/6WWAi1VhbjOJ4FbIXkBtgkK0BNHnYqNKUIPeBPbKyDdzpld5T6wD9SE4AwYjfEDaXFeFzE/doUWuhqwiFsOCwqv2hV2lU/L+sHBscGTxdvSFVoXpAjCZdauMHVic6ndl6U1VBsJCFhTeNUU9IiIEo3qvQYGHAV0AyfC5wNLhKipXuBCjA5wT8WxcM1FMRoBymK44CjAE57hqIazwCfwQdARcXa3UXHuRXVucIjb7jYvNkdxBZg0TBFid7PQTRAtX2xOiXkuMAMqYwkIE848rZFbjyNAmw9bIeweaZ2A5TgC7PnwKkTPtN+cTOrsyN3FEWAjRTAX6sA5ek77gSL6+WHZVQDAIHAjhJtN78aAS3lXAXYIivBOnCdyOAUYB6o0xqsvziry7FLE/Cp20cNcJEjDr8MUmVOVRzkVN+Nd7vZGVXXgiwxtPiRS5WFhz4fEq/zv4AvToMn7vCn3eAAAAABJRU5ErkJggg==",
    Bitmap: "zmdi zmdi-brush",
    Polyline: "zmdi zmdi-gesture",
    Point: "zmdi zmdi-dot-circle-alt",
    Cuboid: "zmdi zmdi-ungroup",  #
    GraphNodes: "zmdi zmdi-grain",
    Cuboid3d: "zmdi zmdi-codepen",
    Pointcloud: "zmdi zmdi-cloud-outline",  # "zmdi zmdi-border-clear"
    MultichannelBitmap: "zmdi zmdi-layers",  # "zmdi zmdi-collection-item"
    Point3d: "zmdi zmdi-filter-center-focus",  # "zmdi zmdi-select-all"
}
class ClassesSelector:
    title = "Classes"
    description = "Select classes that will be used for inference. This classes are defined by the deployed model"
    lock_message = "Select previous step to unlock"

    def __init__(self):
        # Init Step
        self.display_widgets = []
        # -------------------------------- #

        # Init Base Widgets
        self.validator_text = None
        self.button = None
        self.container = None
        self.card = None
        # -------------------------------- #

        # Init Step Widgets
        self.classes_table = None
        # -------------------------------- #

        # Classes
        columns = ["class", "shape"]
        self.classes_table = FastTable(columns=columns, page_size=100, is_selectable=True)
        # self.classes_table = ClassesTable()
        self.classes_table.hide()
        # Add widgets to display ------------ #
        self.display_widgets.extend([self.classes_table])
        # ----------------------------------- #

        # Base Widgets
        self.validator_text = Text("")
        self.validator_text.hide()
        self.button = Button("Select")
        self.display_widgets.extend([self.validator_text, self.button])
        # -------------------------------- #

        # Card Layout
        self.container = Container(self.display_widgets)
        self.card = Card(
            title=self.title,
            description=self.description,
            content=self.container,
            lock_message=self.lock_message,
        )
        # -------------------------------- #

    def lock(self):
        self.card.lock()
        self.classes_table.hide()
        self.validator_text.hide()

    def unlock(self):
        self.card.unlock()
        self.classes_table.show()

    @property
    def widgets_to_disable(self) -> list:
        return [self.classes_table]

    def set_project_meta(self, project_meta) -> None:
        self._update_meta(project_meta)

    def load_from_json(self, data: Dict[str, Any]) -> None:
        if "classes" in data:
            self.set_classes(data["classes"])

    def get_selected_classes(self) -> list:
        return [row.row[0] for row in self.classes_table.get_selected_rows()]

    def set_classes(self, classes) -> None:
        self.classes_table.select_rows_by_value("class", classes)

    def select_all_classes(self) -> None:
        self.classes_table.select_rows(list(range(len(self.classes_table._rows_total))))

    def get_settings(self) -> Dict[str, Any]:
        return {"classes": self.get_selected_classes()}

    def validate_step(self) -> bool:
        if self.classes_table.is_hidden():
            return True

        self.validator_text.hide()
        selected_classes = self.get_selected_classes()
        n_classes = len(selected_classes)

        if n_classes == 0:
            self.validator_text.set(text="Please select at least one class", status="error")
            self.validator_text.show()
            return False

        class_word = "class" if n_classes == 1 else "classes"
        message_parts = [f"Selected {n_classes} {class_word}"]
        status = "success"
        is_valid = True

        self.validator_text.set(text=". ".join(message_parts), status=status)
        self.validator_text.show()
        return is_valid

    def _update_meta(self, project_meta: ProjectMeta) -> None:
        table_data = []
        for obj_class in project_meta.obj_classes:
            table_line = []
            name = obj_class.name
            icon = type_to_zmdi_icon[AnyGeometry]
            icon = type_to_zmdi_icon.get(obj_class.geometry_type, icon)
            shape = obj_class.geometry_type.geometry_name().lower()
            if shape == "graph":
                shape = "graph (keypoints)"
            table_line = [name, shape]
            table_data.append(table_line)
        self._table_data = table_data
        if self._table_data:
            self.classes_table.read_pandas(pd.DataFrame(self._table_data))
        else:
            self.classes_table.clear()
