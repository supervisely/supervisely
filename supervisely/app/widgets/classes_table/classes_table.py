from typing import Optional, List, Dict, Any

from supervisely.app.widgets import Widget
from supervisely.app.widgets.widget import Disableable
import supervisely as sly
from supervisely.geometry.geometry import Geometry
from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.sly_logger import logger


class ClassesTable(Widget):
    class Routes:
        CLASS_SELECTED = "class_selected_cb"

    def __init__(
        self,
        project_meta: Optional[sly.ProjectMeta] = None,
        project_id: Optional[int] = None,
        project_fs: Optional[sly.Project] = None,
        allowed_types: Optional[List[Geometry]] = None,
        selectable: Optional[bool] = True,
        disabled: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        if project_id is not None and project_fs is not None:
            raise ValueError(
                "You can not provide both project_id and project_fs parameters to Classes Table widget."
            )
        self._table_data = []
        self._columns = []
        self._changes_handled = False
        self._global_checkbox = False
        self._checkboxes = []
        self._selectable = selectable
        self._disabled = disabled
        self._loading = False
        self._allowed_types = allowed_types if allowed_types is not None else []
        if project_id is not None:
            self._api = sly.Api()
        else:
            self._api = None
        self._project_id = project_id
        if project_id is not None:
            if project_meta is not None:
                logger.warn(
                    "Both parameters project_id and project_meta were provided to ClassesTable widget. Project meta classes taken from remote project and project_meta parameter is ignored."
                )
            project_meta = sly.ProjectMeta.from_json(self._api.project.get_meta(project_id))
        self._project_fs = project_fs
        if project_fs is not None:
            if project_meta is not None:
                logger.warn(
                    "Both parameters project_fs and project_meta were provided to ClassesTable widget. Project meta classes taken from project_fs.meta and project_meta parameter is ignored."
                )
            project_meta = project_fs.meta
        if project_meta is not None:        
            self._update_meta(project_meta=project_meta)
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

    def _update_meta(self, project_meta: sly.ProjectMeta) -> None:
        columns = ["class", "shape"]
        stats = None
        data_to_show = []
        for obj_class in project_meta.obj_classes:
            if (
                self._allowed_types is None
                or obj_class.geometry_type not in self._allowed_types
            ):
                data_to_show.append(obj_class.to_json())

        if self._project_id is not None:
            stats = self._api.project.get_stats(self._project_id)
            project_info = self._api.project.get_info_by_id(self._project_id)
            if project_info.type == str(sly.ProjectType.IMAGES):
                columns.append("images count")
            elif project_info.type == str(sly.ProjectType.VIDEOS):
                columns.append("videos count")
            elif project_info.type in [
                str(sly.ProjectType.POINT_CLOUDS),
                str(sly.ProjectType.POINT_CLOUD_EPISODES),
            ]:
                columns.append("pointclouds count")
            elif project_info.type == str(sly.ProjectType.VOLUMES):
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

        elif self._project_fs is not None:
            class_items = {}
            class_objects = {}
            for obj_class in project_meta.obj_classes:
                class_items[obj_class.name] = 0
                class_objects[obj_class.name] = 0

            if type(self._project_fs) == sly.Project:
                columns.append("images count")
                for ds in self._project_fs.datasets:
                    ds: sly.Dataset
                    for item_name in ds:
                        item_ann = ds.get_ann(item_name, project_meta)
                        item_class = {}
                        for label in item_ann.labels:
                            label: sly.Label
                            class_objects[label.obj_class.name] += 1
                            item_class[label.obj_class.name] = True
                        for obj_class in project_meta.obj_classes:
                            if obj_class.name in item_class.keys():
                                class_items[obj_class.name] += 1

            elif type(self._project_fs) == sly.VideoProject:
                columns.append("videos count")
                for ds in self._project_fs.datasets:
                    ds: sly.VideoDataset
                    for item_name in ds:
                        item_ann = ds.get_ann(item_name, project_meta)
                        item_ann: sly.VideoAnnotation
                        item_class = {}
                        for video_object in item_ann.objects:
                            class_objects[video_object.obj_class.name] += 1
                            item_class[video_object.obj_class.name] = True
                        for obj_class in project_meta.obj_classes:
                            if obj_class.name in item_class.keys():
                                class_items[obj_class.name] += 1

            elif type(self._project_fs) == sly.PointcloudProject:
                columns.append("pointclouds count")
                for ds in self._project_fs.datasets:
                    ds: sly.PointcloudDataset
                    for item_name in ds:
                        item_ann = ds.get_ann(item_name, project_meta)
                        item_objects = item_ann.get_objects_from_figures()
                        item_class = {}
                        for ptc_object in item_objects:
                            class_objects[ptc_object.obj_class.name] += 1
                            item_class[ptc_object.obj_class.name] = True
                        for obj_class in project_meta.obj_classes:
                            if obj_class.name in item_class.keys():
                                class_items[obj_class.name] += 1

            elif type(self._project_fs) == sly.PointcloudEpisodeProject:
                columns.append("pointclouds count")
                for ds in self._project_fs.datasets:
                    ds: sly.PointcloudEpisodeDataset
                    episode_ann = ds.get_ann(project_meta)
                    for item_name in ds:
                        frame_index = ds.get_frame_idx(item_name)
                        item_objects = episode_ann.get_objects_on_frame(frame_index)
                        item_class = {}
                        for ptc_object in item_objects:
                            class_objects[ptc_object.obj_class.name] += 1
                            item_class[ptc_object.obj_class.name] = True
                        for obj_class in project_meta.obj_classes:
                            if obj_class.name in item_class.keys():
                                class_items[obj_class.name] += 1
                                
            elif type(self._project_fs) == sly.VolumeProject:
                columns.append("volumes count")
                for ds in self._project_fs.datasets:
                    ds: sly.VolumeDataset
                    for item_name in ds:
                        item_ann = ds.get_ann(item_name, project_meta)
                        item_class = {}
                        for volume_object in item_ann.objects:
                            volume_object: sly.VolumeObject
                            class_objects[volume_object.obj_class.name] += 1
                            item_class[volume_object.obj_class.name] = True
                        for obj_class in project_meta.obj_classes:
                            if obj_class.name in item_class.keys():
                                class_items[obj_class.name] += 1
            columns.append("objects count")

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
                        {
                            "name": "CLASS",
                            "data": line["title"],
                            "color": line["color"],
                        },
                        {"name": "SHAPE", "data": line["shape"]},
                    ]
                )
                if "itemsCount" in line.keys():
                    table_line.append(
                        {"name": "ITEMS COUNT", "data": line["itemsCount"]}
                    )
                if "objectsCount" in line.keys():
                    table_line.append(
                        {"name": "OBJECTS COUNT", "data": line["objectsCount"]}
                    )
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

    def read_meta(self, project_meta: sly.ProjectMeta) -> None:
        self.loading = True
        self._project_fs = None
        self._project_id = None
        self.clear_selection()
        self._update_meta(project_meta=project_meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()["checkboxes"] = self._checkboxes
        StateJson()["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def read_project(self, project_fs: sly.Project) -> None:
        self.loading = True
        self._project_fs = project_fs
        self._project_id = None
        self.clear_selection()
        self._update_meta(project_meta=project_fs.meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()["checkboxes"] = self._checkboxes
        StateJson()["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def read_project_from_id(self, project_id: int) -> None:
        self.loading = True
        self._project_fs = None
        self._project_id = project_id
        if self._api is None:
            self._api = sly.Api()
        project_meta = sly.ProjectMeta.from_json(self._api.project.get_meta(project_id))
        self.clear_selection()
        self._update_meta(project_meta=project_meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()["checkboxes"] = self._checkboxes
        StateJson()["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def get_json_data(self) -> Dict[str, Any]:
        return {
            "table_data": self._table_data,
            "columns": self._columns,
            "loading": self._loading,
            "disabled": self._disabled,
            "selectable": self._selectable,
        }

    @property
    def allowed_types(self) -> List[Geometry]:
        return self._allowed_types

    @allowed_types.setter
    def allowed_types(self, value: List[Geometry]):
        self._allowed_types = value
        DataJson()[self.widget_id]["allowed_types"] = self._allowed_types

    def get_json_state(self) -> Dict[str, Any]:
        return {
            "global_checkbox": self._global_checkbox,
            "checkboxes": self._checkboxes,
        }

    def get_selected_classes(self) -> List[str]:
        classes = []
        for i, line in enumerate(self._table_data):
            if StateJson()[self.widget_id]["checkboxes"][i]:
                for col in line:
                    if col["name"] == "CLASS":
                        classes.append(col["data"])
        return classes

    @property
    def loading(self) -> bool:
        return self._loading

    @loading.setter
    def loading(self, value: bool):
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    def clear_selection(self) -> None:
        StateJson()[self.widget_id]["global_checkbox"] = False
        StateJson()[self.widget_id]["checkboxes"] = [False] * len(self._table_data)
        StateJson().send_changes()
