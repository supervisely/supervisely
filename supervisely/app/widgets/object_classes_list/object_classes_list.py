from typing import Union, List
from supervisely.app import DataJson
from supervisely.app.widgets import (
    Widget,
    ObjectClassView,
    Checkbox,
    Grid,
    generate_id,
    Button,
)
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
        self._select_all_btn = None
        self._deselect_all_btn = None
        for obj_class in self._object_classes:
            self._name_to_class[obj_class.name] = obj_class
            self._name_to_view[obj_class.name] = ObjectClassView(
                obj_class, widget_id=generate_id()
            )
            grid_items = list(self._name_to_view.values())
            if self._selectable is True:
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
                current_checkbox = Checkbox(
                    self._name_to_view[obj_class.name],
                    checked=True,
                    widget_id=generate_id(),
                )
                self._name_to_checkbox[obj_class.name] = current_checkbox
                grid_items = list(self._name_to_checkbox.values())

                @self._select_all_btn.click
                def select_all():
                    for k, v in self._name_to_checkbox.items():
                        v: Checkbox
                        v.check()
                    DataJson()[self.widget_id]["num_selected"] = len(
                        self._name_to_checkbox
                    )
                    DataJson().send_changes()

                @self._deselect_all_btn.click
                def deselect_all():
                    for k, v in self._name_to_checkbox.items():
                        v: Checkbox
                        v.uncheck()
                    DataJson()[self.widget_id]["num_selected"] = 0
                    DataJson().send_changes()

                @current_checkbox.value_changed
                def checkbox_changed(is_checked):
                    DataJson()[self.widget_id]["num_selected"] = len(
                        self.get_selected_classes()
                    )
                    DataJson().send_changes()

        self._content = Grid(
            widgets=grid_items,
            columns=self._columns,
            widget_id=generate_id(),
        )
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        res = {"selectable": self._selectable}
        if self._selectable is True:
            res["num_selected"] = len(self.get_selected_classes())
        return res

    def get_selected_classes(self):
        if self._selectable is False:
            raise ValueError(
                "Class selection is disabled, because ObjectClassesList widget was created with argument 'selectable == False'"
            )
        results = []
        for k, v in self._name_to_checkbox.items():
            v: Checkbox
            if v.is_checked():
                results.append(k)
        return results

    def get_json_state(self):
        return None
