from typing import Union, List
from supervisely.app import DataJson
from supervisely.app.widgets import Widget, ObjectClassView, Checkbox, Container
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
            self._name_to_view[obj_class.name] = ObjectClassView(obj_class)
            self._name_to_checkbox[obj_class.name] = Checkbox(self._name_to_view[obj_class.name], checked=True)
            
        if self._columns < 1:
            raise ValueError(f"columns ({self._columns}) < 1")
        if self._columns > len(self._object_classes):
            self._columns = len(self._object_classes)
            
        self._content = None
        if self._columns == 1:
            self._content = Container(direction="vertical", widgets=list(self._name_to_checkbox.values()))
        # else:
            # rows = []
            # for 
        super().__init__(widget_id=widget_id, file_path=__file__)

    def _create_widgets():
        
    
    def get_json_data(self):
        
        
        res = self._obj_class.to_json()
        res["icon"] = None
        res["icon8"] = None
        if self._show_shape_icon is True:
            res["icon"] = type_to_zmdi_icon.get(self._obj_class.geometry_type)
            res["icon8"] = type_to_icons8_icon.get(self._obj_class.geometry_type)
        res["shape_text"] = None
        if self._show_shape_text is True:
            res["shape_text"] = type_to_shape_text.get(
                self._obj_class.geometry_type
            ).upper()
        return res

    def get_json_state(self):
        return None
