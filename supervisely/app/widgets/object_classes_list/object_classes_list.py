from typing import Union, List
from supervisely.app import DataJson
from supervisely.app.widgets import Widget, ObjectClassView, Checkbox, Grid, generate_id
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection


class ObjectClassesList(Widget):
    def __init__(
        self,
        object_classes: Union[ObjClassCollection, List[ObjClass]],
        selectable: bool = False,
        columns: int = 1,  # 1 means vertical layout
        widget_id: str = None,
    ):
        self._object_classes = object_classes
        self._selectable = selectable
        self._columns = columns

        if type(object_classes) is list:
            self._object_classes = ObjClassCollection(self._object_classes)

        self._name_to_class = {}
        self._name_to_view = {}
        self._name_to_checkbox = {}
        for obj_class in self._object_classes:
            self._name_to_class[obj_class.name] = obj_class
            self._name_to_view[obj_class.name] = ObjectClassView(
                obj_class, widget_id=generate_id()
            )
            self._name_to_checkbox[obj_class.name] = Checkbox(
                self._name_to_view[obj_class.name],
                checked=True,
                widget_id=generate_id(),
            )

        self._content = Grid(
            widgets=list(self._name_to_checkbox.values()),
            columns=self._columns,
            widget_id=generate_id(),
        )
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return None

    def get_json_state(self):
        return None
