from typing import Any, Callable, Dict, List, Optional

import supervisely as sly
from supervisely.app import DataJson
from supervisely.app.content import StateJson
from supervisely.app.widgets import Widget
from supervisely.sly_logger import logger
from supervisely.annotation.tag_meta import TagValueType


class TagsTable(Widget):
    """TagsTable widget in Supervisely allows users to display all tags from given project in a table format.

    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/tables/TagsTable>`_
        (including screenshots and examples).

    :param project_meta: Project meta object from which tags will be taken.
    :type project_meta: sly.ProjectMeta
    :param project_id: Project id from which tags will be taken.
    :type project_id: int
    :param project_fs: Project object from which tags will be taken.
    :type project_fs: sly.Project
    :param allowed_types: List of allowed value types to be displayed in table.
    :type allowed_types: List[TagValueType]
    :param selectable: If True, user can select tags from table.
    :type selectable: bool
    :param disabled: If True, the elements in the table will be disabled.
    :type disabled: bool
    :param widget_id: Unique widget identifier.
    :type widget_id: str
    :raises ValueError: If both project_id and project_fs parameters are provided.

    :Usage example:
    .. code-block:: python

        from supervisely.app.widgets import TagsTable

        tags_table = TagsTable(project_id=123, selectable=True)
    """

    class Routes:
        TAG_SELECTED = "tag_selected_cb"

    def __init__(
        self,
        project_meta: Optional[sly.ProjectMeta] = None,
        project_id: Optional[int] = None,
        project_fs: Optional[sly.Project] = None,
        allowed_types: Optional[List[TagValueType]] = None,
        selectable: Optional[bool] = True,
        disabled: Optional[bool] = False,
        widget_id: Optional[str] = None,
    ):
        if project_id is not None and project_fs is not None:
            raise ValueError(
                "You can not provide both project_id and project_fs parameters to Tags Table widget."
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
                    "Both parameters project_id and project_meta were provided to TagsTable widget. Project meta tags taken from remote project and project_meta parameter is ignored."
                )
            project_meta = sly.ProjectMeta.from_json(self._api.project.get_meta(project_id))
        self._project_fs = project_fs
        if project_fs is not None:
            if project_meta is not None:
                logger.warn(
                    "Both parameters project_fs and project_meta were provided to TagsTable widget. Project meta tags taken from project_fs.meta and project_meta parameter is ignored."
                )
            project_meta = project_fs.meta

        self._project_meta = project_meta
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
        route_path = self.get_route_path(TagsTable.Routes.TAG_SELECTED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _value_changed():
            res = self.get_selected_tags()
            func(res)

        return _value_changed

    def _update_meta(self, project_meta: sly.ProjectMeta) -> None:
        columns = ["tag", "type", "applicable to", "possible values"]
        stats = None
        data_to_show = []
        for tag_meta in project_meta.tag_metas:
            if len(self._allowed_types) == 0 or tag_meta.value_type in self._allowed_types:
                data_to_show.append(tag_meta.to_json())

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

            tag_items = {}
            for item in stats["imageTags"]["items"]:
                tag_items[item["tagMeta"]["name"]] = item["total"]

            tag_objects = {}
            for item in stats["objectTags"]["items"]:
                tag_objects[item["tagMeta"]["name"]] = item["total"]
            for tag_meta in data_to_show:
                tag_meta["itemsCount"] = tag_items[tag_meta["name"]]
                tag_meta["objectsCount"] = tag_objects[tag_meta["name"]]

        elif self._project_fs is not None:
            project_stats = self._project_fs.get_tags_stats()

            if type(self._project_fs) == sly.Project:
                columns.extend(["images count", "labels count"])

            elif type(self._project_fs) == sly.VideoProject:
                columns.extend(["videos count", "objects count", "figures count"])

            elif type(self._project_fs) in [sly.PointcloudProject, sly.PointcloudEpisodeProject]:
                columns.extend(["pointclouds count", "objects count", "figures count"])

            elif type(self._project_fs) == sly.VolumeProject:
                columns.extend(["volumes count", "objects count", "figures count"])

            for tag_meta in data_to_show:
                tag_meta["itemsCount"] = project_stats["items_count"][tag_meta["title"]]
                tag_meta["objectsCount"] = project_stats["objects_count"][tag_meta["title"]]
                if type(self._project_fs) != sly.Project:
                    tag_meta["figuresCount"] = project_stats["figures_count"][tag_meta["title"]]

        columns = [col.upper() for col in columns]
        if data_to_show:
            table_data = []
            if self._project_id is not None or self._project_fs is not None:
                data_to_show = sorted(
                    data_to_show, key=lambda line: line["objectsCount"], reverse=True
                )
            for line in data_to_show:
                table_line = []
                table_line.extend(
                    [
                        {
                            "name": "TAG",
                            "data": line["name"],
                            "color": line["color"],
                        },
                        {"name": "TYPE", "data": line["value_type"]},
                        {"name": "APPLICABLE TO", "data": line["applicable_type"]},
                        {"name": "POSSIBLE VALUES", "data": line.get("values", "-")},
                    ]
                )
                if "itemsCount" in line.keys():
                    table_line.append({"name": "ITEMS COUNT", "data": line["itemsCount"]})
                if "objectsCount" in line.keys():
                    table_line.append({"name": "OBJECTS COUNT", "data": line["objectsCount"]})
                # if "figuresCount" in line.keys():
                # table_line.append({"name": "FIGURES COUNT", "data": line["figuresCount"]})
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

        :param project_meta: Project meta object from which tags will be taken.
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

        :param project_fs: Project object from which tags will be taken.
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

    def read_project_from_id(self, project_id: int) -> None:
        """Read remote project by id and update table data.

        :param project_id: Project id from which tags will be taken.
        :type project_id: int
        """
        self.loading = True
        self._project_fs = None
        self._project_id = project_id
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
            - selectable: If True, user can select tags from table.

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
    def allowed_types(self) -> List[TagValueType]:
        """Returns list of allowed tag value types to be displayed in table.

        :return: List of allowed tag value types to be displayed in table.
        :rtype: List[TagValueType]
        """
        return self._allowed_types

    @property
    def project_id(self) -> int:
        """Returns project id from which tags was taken.

        :return: Project id from which tags was taken.
        :rtype: int
        """
        return self._project_id

    @property
    def project_fs(self) -> int:
        """Returns project object from which tags was taken.

        :return: Project object from which tags was taken.
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
        """Returns project meta object from which tags was taken.

        :return: Project meta object from which tags was taken.
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

    def get_selected_tags(self) -> List[str]:
        """Returns list of selected tags.

        :return: List of selected tags.
        :rtype: List[str]
        """
        tags = []
        for i, line in enumerate(self._table_data):
            checkboxes = StateJson()[self.widget_id]["checkboxes"]
            if len(checkboxes) == 0:
                checkboxes = [False] * len(self._table_data)
            if checkboxes[i]:
                for col in line:
                    if col["name"] == "TAG":
                        tags.append(col["data"])
        return tags

    def clear_selection(self) -> None:
        """Clears selection of tags."""
        self._global_checkbox = False
        self._checkboxes = [False] * len(self._table_data)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()

    def set_project_meta(self, project_meta: sly.ProjectMeta) -> None:
        """Sets project meta object from which tags will be taken.

        :param project_meta: Project meta object from which tags will be taken.
        :type project_meta: sly.ProjectMeta
        """
        self._update_meta(project_meta)
        self._project_meta = project_meta
        self.update_data()
        DataJson().send_changes()

    def select_tags(self, tags: List[str]) -> None:
        """Selects tags in the table from given list.

        :param tags: List of tags to be selected.
        :type tags: List[str]
        """
        self._global_checkbox = False
        self._checkboxes = [False] * len(self._table_data)

        project_tags = []
        for i, line in enumerate(self._table_data):
            for col in line:
                if col["name"] == "TAG":
                    project_tags.append(col["data"])

        for i, cls_name in enumerate(project_tags):
            if cls_name in tags:
                self._checkboxes[i] = True
        self._global_checkbox = all(self._checkboxes)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()

    def select_all(self) -> None:
        """Selects all tags in the table."""
        self._global_checkbox = True
        self._checkboxes = [True] * len(self._table_data)
        StateJson()[self.widget_id]["global_checkbox"] = self._global_checkbox
        StateJson()[self.widget_id]["checkboxes"] = self._checkboxes
        StateJson().send_changes()
