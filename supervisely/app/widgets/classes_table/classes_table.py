from typing import Optional, List

from supervisely.app.widgets import Widget
from supervisely.app.widgets.widget import Disableable
from supervisely import ProjectMeta, ProjectType, Api, Project
from supervisely.geometry.geometry import Geometry
from supervisely.app import DataJson
from supervisely.app.content import StateJson


class ClassesTable(Widget):
    class Routes:
        CLASS_SELECTED = "class_selected_cb"

    def __init__(
        self,
        project_meta: Optional[ProjectMeta] = None,
        project_id: Optional[int] = None,
        project_fs: Optional[Project] = None,
        allowed_types: Optional[List[Geometry]] = None,
        disabled: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        self._table_data = []
        self._columns = []
        self._changes_handled = False
        self._global_checkbox = False
        self._checkboxes = []
        self._allowed_types = allowed_types if allowed_types is not None else []
        self._api = Api()
        self._project_id = project_id
        self._disabled = disabled
        self._update_meta(project_meta=project_meta)

        self._loading = False

        super().__init__(widget_id=widget_id, file_path=__file__)

    def value_changed(self, func):
        route_path = self.get_route_path(ClassesTable.Routes.CLASS_SELECTED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_classes()
            func(res)

        return _value_changed

    def _update_meta(self, project_meta: Optional[ProjectMeta] = None) -> None:
        if project_meta is None:
            self._table_data = []
            self._columns = []
            self._checkboxes = []
            self._global_checkbox = False
            return
        columns = ["class", "shape"]
        stats = None
        data_to_show = []
        for obj_class in project_meta.obj_classes:
            if self._allowed_types is None or obj_class.geometry_type not in self._allowed_types:
                data_to_show.append(obj_class.to_json())

        if self._project_id is not None:
            stats = self._api.project.get_stats(self._project_id)
            project_info = self._api.project.get_info_by_id(self._project_id)
            if project_info.type == str(ProjectType.IMAGES):
                columns.append("images count")
            elif project_info.type == str(ProjectType.VIDEOS):
                columns.append("videos count")
            elif project_info.type in [
                str(ProjectType.POINT_CLOUDS),
                str(ProjectType.POINT_CLOUD_EPISODES),
            ]:
                columns.append("pointclouds count")
            elif project_info.type == str(ProjectType.VOLUMES):
                columns.append("volumes count")
            columns.append("objects count")

            class_items = {}
            for item in stats["images"]["objectClasses"]:
                class_items[item["objectClass"]["name"]] = item["total"]

            class_objects = {}
            for item in stats["objects"]["items"]:
                class_objects[item["objectClass"]["name"]] = item["total"]
            for obj_class in data_to_show:
                obj_class["itemsCount"] = class_items[obj_class["title"]]
                obj_class["objectsCount"] = class_objects[obj_class["title"]]

        columns = [col.upper() for col in columns]
        if data_to_show:
            table_data = []
            for line in data_to_show:
                table_line = []
                table_line.extend(
                    [
                        {"name": "CLASS", "data": line["title"], "color": line["color"]},
                        {"name": "SHAPE", "data": line["shape"]},
                    ]
                )
                if "itemsCount" in line.keys():
                    table_line.append({"name": "ITEMS COUNT", "data": line["itemsCount"]})
                if "objectsCount" in line.keys():
                    table_line.append({"name": "OBJECTS COUNT", "data": line["objectsCount"]})
                table_data.append(table_line)
            self._table_data = table_data
            self._columns = columns
            self._checkboxes = [False] * len(table_data)
            self._global_checkbox = False
        else:
            self._table_data = []
            self._columns = []
            self._checkboxes = []
            self._global_checkbox = False

    def read_meta(self, project_meta: ProjectMeta) -> None:
        self._update_meta(project_meta=project_meta)
        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        self.clear_selection()

    def get_json_data(self):
        return {
            "table_data": self._table_data,
            "columns": self._columns,
            "loading": self._loading,
            "disabled": self._disabled,
        }

    @property
    def project_id(self) -> int:
        return self._project_id

    @project_id.setter
    def project_id(self, value: int):
        self._project_id = value
        DataJson()[self.widget_id]["project_id"] = self._project_id

    @property
    def allowed_types(self) -> List[Geometry]:
        return self._allowed_types

    @allowed_types.setter
    def allowed_types(self, value: List[Geometry]):
        self._allowed_types = value
        DataJson()[self.widget_id]["allowed_types"] = self._allowed_types

    def get_json_state(self):
        return {"global_checkbox": self._global_checkbox, "checkboxes": self._checkboxes}

    def get_selected_classes(self) -> List[str]:
        classes = []
        for i, line in enumerate(self._table_data):
            if StateJson()[self.widget_id]["checkboxes"][i]:
                for col in line:
                    if col["name"] == "CLASS":
                        classes.append(col["data"])
        return classes

    @property
    def loading(self):
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value: bool):
        self._disabled = value
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def clear_selection(self):
        StateJson()[self.widget_id]["global_checkbox"] = False
        StateJson()[self.widget_id]["checkboxes"] = [False] * len(self._table_data)
        StateJson().send_changes()
