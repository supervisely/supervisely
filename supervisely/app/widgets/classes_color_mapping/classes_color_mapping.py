from typing import List, Optional
from supervisely.imaging.color import rgb2hex, hex2rgb
from supervisely.app.widgets import Widget
from supervisely.app import StateJson, DataJson
from supervisely.app.widgets.classes_mapping.classes_mapping import type_to_shape_text


class ClassesColorMapping(Widget):
    def __init__(self, classes=[], greyscale=False, widget_id=None):
        self._classes = classes
        self._greyscale = greyscale
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "classes": [
                {
                    **cls.to_json(),
                    "shape_text": type_to_shape_text.get(cls.geometry_type).upper(),
                    "default_value": rgb2hex(cls.color),
                }
                for cls in self._classes
            ]
        }

    def get_json_state(self):
        return {
            "classes_values": [
                {
                    "value": rgb2hex(cls.color),
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
                {"value": rgb2hex(cls.color), "default": False, "ignore": True, "selected": False},
            )
            new_mapping_values.append(value)
        StateJson()[self.widget_id]["classes_values"] = new_mapping_values
        StateJson().send_changes()

    def set_colors(self, classes_colors: List[List[int]]):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        for idx, cls in enumerate(self._classes):
            if classes_colors[idx][0] == "#":
                hex_color = classes_colors[idx]
            else:
                hex_color = rgb2hex(classes_colors[idx])
            classes_values[idx]["value"] = hex_color
            classes_values[idx]["default"] = tuple(cls.color) == tuple(hex2rgb(hex_color))
            classes_values[idx]["ignore"] = False
        StateJson()[self.widget_id]["classes_values"] = classes_values
        StateJson().send_changes()

    def get_classes(self):
        return self._classes

    def get_selected_classes_original(self):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        return [cls for idx, cls in enumerate(self._classes) if classes_values[idx]["selected"]]

    def get_selected_classes_edited(self):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        selected_classes = [
            cls for idx, cls in enumerate(self._classes) if classes_values[idx]["selected"]
        ]
        mapping = self.get_mapping()
        new_classes = [
            cls.clone(color=hex2rgb(mapping[cls.name]["value"]))
            for cls in selected_classes
            if cls.name in mapping
        ]
        return new_classes

    def get_mapping(self):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        if len(classes_values) != len(self._classes):
            self.update_state()
            return self.get_mapping()
        mapping = {cls.name: classes_values[idx] for idx, cls in enumerate(self._classes)}
        return mapping

    def set_default(self):
        self.update_state()
        StateJson().send_changes()

    def select(self, classes: List[str]):
        classes_values = StateJson()[self.widget_id]["classes_values"]
        for idx, obj_class in enumerate(self._classes):
            classes_values[idx]["selected"] = obj_class.name in classes
        StateJson()[self.widget_id]["classes_values"] = classes_values
        StateJson().send_changes()
