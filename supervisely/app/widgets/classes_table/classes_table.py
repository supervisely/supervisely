from typing import Any, Callable, Dict, List, Optional

import supervisely as sly
from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.geometry import Geometry
from supervisely.geometry.point_3d import Point3d
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.sly_logger import logger

type_to_zmdi_icon = {
    sly.AnyGeometry: "zmdi zmdi-shape",
    sly.Rectangle: "zmdi zmdi-crop-din",  # "zmdi zmdi-square-o"
    # sly.Polygon: "icons8-polygon",  # "zmdi zmdi-edit"
    sly.Polygon: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmJLR0QA/wD/AP+gvaeTAAAB6klEQVRYhe2Wuy8EURTGf+u5VESNXq2yhYZCoeBv8RcI1i6NVUpsoVCKkHjUGlFTiYb1mFmh2MiKjVXMudmb3cPOzB0VXzKZm5k53/nmvO6Ff4RHD5AD7gFP1l3Kd11AHvCBEpAVW2esAvWmK6t8l1O+W0lCQEnIJoAZxUnzNQNkZF36jrQjgoA+uaciCgc9VaExBOyh/6WWAi1VhbjOJ4FbIXkBtgkK0BNHnYqNKUIPeBPbKyDdzpld5T6wD9SE4AwYjfEDaXFeFzE/doUWuhqwiFsOCwqv2hV2lU/L+sHBscGTxdvSFVoXpAjCZdauMHVic6ndl6U1VBsJCFhTeNUU9IiIEo3qvQYGHAV0AyfC5wNLhKipXuBCjA5wT8WxcM1FMRoBymK44CjAE57hqIazwCfwQdARcXa3UXHuRXVucIjb7jYvNkdxBZg0TBFid7PQTRAtX2xOiXkuMAMqYwkIE848rZFbjyNAmw9bIeweaZ2A5TgC7PnwKkTPtN+cTOrsyN3FEWAjRTAX6sA5ek77gSL6+WHZVQDAIHAjhJtN78aAS3lXAXYIivBOnCdyOAUYB6o0xqsvziry7FLE/Cp20cNcJEjDr8MUmVOVRzkVN+Nd7vZGVXXgiwxtPiRS5WFhz4fEq/zv4AvToMn7vCn3eAAAAABJRU5ErkJggg==",
    sly.Bitmap: "zmdi zmdi-brush",
    sly.Polyline: "zmdi zmdi-gesture",
    sly.Point: "zmdi zmdi-dot-circle-alt",
    sly.Cuboid: "zmdi zmdi-ungroup",  #
    sly.GraphNodes: "zmdi zmdi-grain",
    Cuboid3d: "zmdi zmdi-codepen",
    Pointcloud: "zmdi zmdi-cloud-outline",  # "zmdi zmdi-border-clear"
    sly.MultichannelBitmap: "zmdi zmdi-layers",  # "zmdi zmdi-collection-item"
    Point3d: "zmdi zmdi-filter-center-focus",  # "zmdi zmdi-select-all"
}


class ClassesTable(Widget):
    """ClassesTable widget in Supervisely allows users to display all classes from given project in a table format.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/tables/classestable>`_
        (including screenshots and examples).

    :param project_meta: Project meta object from which classes will be taken.
    :type project_meta: sly.ProjectMeta
    :param project_id: Project id from which classes will be taken.
    :type project_id: int
    :param project_fs: Project object from which classes will be taken.
    :type project_fs: sly.Project
    :param allowed_types: List of allowed geometry types to be displayed in table.
    :type allowed_types: List[Geometry]
    :param selectable: If True, user can select classes from table.
    :type selectable: bool
    :param disabled: If True, the elements in the table will be disabled.
    :type disabled: bool
    :param widget_id: Unique widget identifier.
    :type widget_id: str
    :raises ValueError: If both project_id and project_fs parameters are provided.

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import ClassesTable

        classes_table = ClassesTable(project_id=123, selectable=True)
    """

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
        dataset_ids: Optional[List[int]] = None,
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
        self._selection_disabled = disabled
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

        self._project_meta = project_meta
        self._dataset_ids = dataset_ids
        if project_meta is not None:
            self._update_meta(project_meta=project_meta)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def value_changed(self, func: Callable[[List[str]], Any]) -> Callable[[], None]:
        """Decorator for the function to be called when the value of the widget changes.

        :param func: Function to be called when the value of the widget changes.
        :type func: Callable[[List[str]], Any]
        :return: Decorated function.
        :rtype: Callable[[], None]
        """
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
            if len(self._allowed_types) == 0 or obj_class.geometry_type in self._allowed_types:
                data_to_show.append(obj_class.to_json())

        if self._project_id is not None:
            stats = self._api.project.get_stats(self._project_id)
            project_info = self._api.project.get_info_by_id(self._project_id)
            if project_info.type == str(sly.ProjectType.IMAGES):
                columns.extend(["images count", "labels count"])
            elif project_info.type == str(sly.ProjectType.VIDEOS):
                columns.extend(["videos count", "figures count"])
            elif project_info.type in [
                str(sly.ProjectType.POINT_CLOUDS),
                str(sly.ProjectType.POINT_CLOUD_EPISODES),
            ]:
                columns.extend(["pointclouds count", "figures count"])
            elif project_info.type == str(sly.ProjectType.VOLUMES):
                columns.extend(["volumes count", "figures count"])

            class_items = {}
            for item in stats["images"]["objectClasses"]:
                if self._dataset_ids is not None:
                    items_count = 0
                    for cls_dataset_stats in item["datasets"]:
                        if cls_dataset_stats["id"] in self._dataset_ids:
                            items_count += cls_dataset_stats["count"]
                    class_items[item["objectClass"]["name"]] = items_count
                else:
                    class_items[item["objectClass"]["name"]] = item["total"]

            class_objects = {}
            for item in stats["objects"]["items"]:
                if self._dataset_ids is not None:
                    items_count = 0
                    for cls_dataset_stats in item["datasets"]:
                        if cls_dataset_stats["id"] in self._dataset_ids:
                            items_count += cls_dataset_stats["count"]
                    class_objects[item["objectClass"]["name"]] = items_count
                else:
                    class_objects[item["objectClass"]["name"]] = item["total"]
            for obj_class in data_to_show:
                obj_class["itemsCount"] = class_items[obj_class["title"]]
                obj_class["objectsCount"] = class_objects[obj_class["title"]]

        elif self._project_fs is not None:
            project_stats = self._project_fs.get_classes_stats()

            if type(self._project_fs) == sly.Project:
                columns.extend(["images count", "labels count"])

            elif type(self._project_fs) == sly.VideoProject:
                columns.extend(["videos count", "objects count", "figures count"])

            elif type(self._project_fs) in [sly.PointcloudProject, sly.PointcloudEpisodeProject]:
                columns.extend(["pointclouds count", "objects count", "figures count"])

            elif type(self._project_fs) == sly.VolumeProject:
                columns.extend(["volumes count", "objects count", "figures count"])

            for obj_class in data_to_show:
                obj_class["itemsCount"] = project_stats["items_count"][obj_class["title"]]
                obj_class["objectsCount"] = project_stats["objects_count"][obj_class["title"]]
                if type(self._project_fs) != sly.Project:
                    obj_class["figuresCount"] = project_stats["figures_count"][obj_class["title"]]

        columns = [col.upper() for col in columns]
        if data_to_show:
            table_data = []
            if self._project_id is not None or self._project_fs is not None:
                data_to_show = sorted(
                    data_to_show, key=lambda line: line["objectsCount"], reverse=True
                )
            for line in data_to_show:
                table_line = []
                icon = type_to_zmdi_icon[sly.AnyGeometry]
                for geo_type, icon_text in type_to_zmdi_icon.items():
                    geo_type: Geometry
                    if geo_type.geometry_name() == line["shape"]:
                        icon = icon_text
                        break
                if line["shape"] == "graph":
                    line["shape"] = "graph (keypoints)"
                table_line.extend(
                    [
                        {
                            "name": "CLASS",
                            "data": line["title"],
                            "color": line["color"],
                        },
                        {"name": "SHAPE", "data": line["shape"], "icon": icon},
                    ]
                )
                if "itemsCount" in line.keys():
                    table_line.append({"name": "ITEMS COUNT", "data": line["itemsCount"]})
                if "objectsCount" in line.keys():
                    table_line.append({"name": "OBJECTS COUNT", "data": line["objectsCount"]})
                if "figuresCount" in line.keys():
                    table_line.append({"name": "FIGURES COUNT", "data": line["figuresCount"]})
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
        """Read project meta and update table data.

        :param project_meta: Project meta object from which classes will be taken.
        :type project_meta: sly.ProjectMeta
        """
        self.loading = True
        self._project_fs = None
        self._project_id = None
        self._project_meta = project_meta
        self.clear_selection()
        self._update_meta(project_meta=project_meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def read_project(self, project_fs: sly.Project) -> None:
        """Read local project and update table data.

        :param project_fs: Project object from which classes will be taken.
        :type project_fs: sly.Project
        """
        self.loading = True
        self._project_fs = project_fs
        self._project_id = None
        self._project_meta = project_fs.meta
        self.clear_selection()
        self._update_meta(project_meta=project_fs.meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def read_project_from_id(
        self, project_id: int, dataset_ids: Optional[List[int]] = None
    ) -> None:
        """Read remote project by id and update table data.

        :param project_id: Project id from which classes will be taken.
        :type project_id: int
        :param dataset_ids: List of dataset ids to filter classes.
        :type dataset_ids: Optional[List[int]]
        """
        self.loading = True
        self._project_fs = None
        self._project_id = project_id
        if dataset_ids is not None:
            self._dataset_ids = dataset_ids
        if self._api is None:
            self._api = sly.Api()
        project_meta = sly.ProjectMeta.from_json(self._api.project.get_meta(project_id))
        self._project_meta = project_meta
        self.clear_selection()
        self._update_meta(project_meta=project_meta)

        DataJson()[self.widget_id]["table_data"] = self._table_data
        DataJson()[self.widget_id]["columns"] = self._columns
        DataJson().send_changes()
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson().send_changes()
        self.loading = False

    def get_json_data(self) -> Dict[str, Any]:
        """Returns dictionary with widget data, which defines the appearance and behavior of the widget.

        Dictionary contains the following fields:
            - table_data: List of dictionaries with table data.
            - columns: List of column names.
            - loading: If True, the widget is in loading state.
            - disabled: If True, the elements in the table will be disabled.
            - selectable: If True, user can select classes from table.

        :return: Dictionary with widget data.
        :rtype: Dict[str, Any]
        """
        return {
            "table_data": self._table_data,
            "columns": self._columns,
            "loading": self._loading,
            "disabled": self._selection_disabled,
            "selectable": self._selectable,
        }

    @property
    def allowed_types(self) -> List[Geometry]:
        """Returns list of allowed geometry types to be displayed in table.

        :return: List of allowed geometry types to be displayed in table.
        :rtype: List[Geometry]
        """
        return self._allowed_types

    @property
    def project_id(self) -> int:
        """Returns project id from which classes was taken.

        :return: Project id from which classes was taken.
        :rtype: int
        """
        return self._project_id

    @property
    def project_fs(self) -> int:
        """Returns project object from which classes was taken.

        :return: Project object from which classes was taken.
        :rtype: sly.Project
        """
        return self._project_fs

    @property
    def loading(self) -> bool:
        """Returns True if the widget is in loading state.

        :return: True if the widget is in loading state.
        :rtype: bool
        """
        return self._loading

    @property
    def project_meta(self) -> bool:
        """Returns project meta object from which classes was taken.

        :return: Project meta object from which classes was taken.
        :rtype: sly.ProjectMeta
        """
        return self._project_meta

    @loading.setter
    def loading(self, value: bool) -> None:
        """Sets loading state of the widget.

        :param value: Loading state of the widget.
        :type value: bool
        """
        self._loading = value
        DataJson()[self.widget_id]["loading"] = self._loading
        DataJson().send_changes()

    def get_json_state(self) -> Dict[str, Any]:
        """Returns dictionary with widget state.

        Dictionary contains the following fields:
            - global_checkbox: State of global checkbox.
            - checkboxes: List of checkboxes states.

        :return: Dictionary with widget state.
        :rtype: Dict[str, Any]
        """
        return {
            "global_checkbox": self._global_checkbox,
            "checkboxes": self._checkboxes,
        }

    def get_selected_classes(self) -> List[str]:
        """Returns list of selected classes.

        :return: List of selected classes.
        :rtype: List[str]
        """
        classes = []
        checkboxes = StateJson()[self.widget_id]["checkboxes"]
        for i, line in enumerate(self._table_data):
            if len(checkboxes) == 0:
                checkboxes = [False] * len(self._table_data)
            if i >= len(checkboxes):
                continue
            if checkboxes[i]:
                for col in line:
                    if col["name"] == "CLASS":
                        classes.append(col["data"])
        return classes

    def clear_selection(self) -> None:
        """Clears selection of classes."""
        self._global_checkbox = False
        self._checkboxes = [False] * len(self._table_data)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()

    def set_project_meta(self, project_meta: sly.ProjectMeta) -> None:
        """Sets project meta object from which classes will be taken.

        :param project_meta: Project meta object from which classes will be taken.
        :type project_meta: sly.ProjectMeta
        """
        self._update_meta(project_meta)
        self._project_meta = project_meta
        self.update_data()
        DataJson().send_changes()

    def select_classes(self, classes: List[str]) -> None:
        """Selects classes in the table from given list.

        :param classes: List of classes to be selected.
        :type classes: List[str]
        """
        self._global_checkbox = False
        self._checkboxes = [False] * len(self._table_data)

        project_classes = []
        for i, line in enumerate(self._table_data):
            for col in line:
                if col["name"] == "CLASS":
                    project_classes.append(col["data"])

        for i, cls_name in enumerate(project_classes):
            if cls_name in classes:
                self._checkboxes[i] = True
        self._global_checkbox = all(self._checkboxes)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()

    def select_all(self) -> None:
        """Selects all classes in the table."""
        self._global_checkbox = True
        self._checkboxes = [True] * len(self._table_data)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()

    def set_dataset_ids(self, dataset_ids: List[int]) -> None:
        """Sets dataset ids to filter classes.

        :param dataset_ids: List of dataset ids to filter classes.
        :type dataset_ids: List[int]
        """
        selected_classes = self.get_selected_classes()
        self._dataset_ids = dataset_ids
        self._update_meta(self._project_meta)
        self.update_data()
        self.select_classes(selected_classes)
