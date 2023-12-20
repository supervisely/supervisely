from typing import Optional, Union, List

from supervisely import ObjClass, ObjClassCollection
from supervisely.app import StateJson
from supervisely.app.widgets import Widget


class ClassesMappingPreview(Widget):
    def __init__(
        self,
        classes: Optional[Union[List[ObjClass], ObjClassCollection]] = [],
        mapping: Optional[dict] = {},
        max_height: str = "128px",
        widget_id: Optional[str] = None,
    ):
        self._classes = classes
        self._mapping = mapping
        self._max_height = max_height
        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "max_height": self._max_height,
        }

    def get_json_state(self):
        return {
            "mapping": [
                {"class": cls.to_json(), "value": self._mapping.get(cls.name, "")}
                for cls in self._classes
            ]
        }

    def set(self, classes: Union[List[ObjClass], ObjClassCollection], mapping: dict):
        self._mapping = {
            k: v["value"]
            for k, v in mapping.items()
            if k in [obj_class.name for obj_class in classes]
        }
        self._classes = [obj_class for obj_class in classes if obj_class.name in self._mapping]
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

    def set_mapping(self, mapping: dict):
        if self._classes is None:
            raise ValueError("Classes are not set")
        self._mapping = {
            k: v["value"]
            for k, v in mapping.items()
            if k in [obj_class.name for obj_class in self._classes]
        }
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

    def get_classes(self):
        return self._classes

    def get_mapping(self):
        return self._mapping
