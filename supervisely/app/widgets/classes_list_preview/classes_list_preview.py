from typing import Optional, Union, List

from supervisely import ObjClass, ObjClassCollection
from supervisely.app import StateJson
from supervisely.app.widgets import Widget, ObjectClassView


class ClassesListPreview(Widget):
    def __init__(
        self,
        classes: Optional[Union[List[ObjClass], ObjClassCollection]] = [],
        max_height: str = "128px",
        empty_text: str = None,
        show_shape_title: bool = True,
        show_shape_icon: bool = True,
        widget_id: Optional[str] = None,
    ):
        self._classes = [
            ObjectClassView(cls, show_shape_title, show_shape_icon).get_json_data()
            for cls in classes
        ]
        self._max_height = max_height
        self._empty_text = empty_text
        self._show_shape_title = show_shape_title
        self._show_shape_icon = show_shape_icon

        super().__init__(widget_id=widget_id, file_path=__file__)

    def get_json_data(self):
        return {
            "max_height": self._max_height,
        }

    def get_json_state(self):
        return {"classes": [cls for cls in self._classes]}

    def set(
        self,
        classes: Union[List[ObjClass], ObjClassCollection],
        show_shape_title: bool = True,
        show_shape_icon: bool = True,
    ):
        self._classes = [
            ObjectClassView(cls, show_shape_title, show_shape_icon).get_json_data()
            for cls in classes
        ]
        StateJson()[self.widget_id] = self.get_json_state()
        StateJson().send_changes()

    def get(self):
        return self._classes
