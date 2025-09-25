import json
from datetime import datetime
from typing import Callable, List, Literal, Optional

import pandas as pd

from supervisely import env
from supervisely.api.api import Api
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.module_api import ApiField
from supervisely.api.project_api import ProjectInfo
from supervisely.app.content import StateJson
from supervisely.app.widgets import (
    Container,
    DatasetThumbnail,
    Empty,
    FastTable,
    ProjectThumbnail,
    Select,
    Text,
    Widget,
)
from supervisely.project import ProjectType


class TeamWorkspaceSelect(Widget):
    class Routes:
        VALUE_CHANGED = "value_changed"

    def __init__(
        self,
        show_label: bool = True,
        direction: Literal["horizontal", "vertical"] = "vertical",
        default_team_id: Optional[int] = None,
        default_workspace_id: Optional[int] = None,
        size: Literal["large", "small", "mini"] = None,
        selectors_style: str = "",
        widget_id=None,
    ):
        self._api = Api()
        self._direction = direction
        self._show_label = show_label
        self._size = size
        self._team_id = default_team_id
        self._workspace_id = default_workspace_id
        if self._team_id is None and self._workspace_id is None:
            self._team_id = env.team_id()
            self._workspace_id = self._api.workspace.get_list(self._team_id)[0].id
        elif self._team_id is None and self._workspace_id is not None:
            self._team_id = self._api.workspace.get_info_by_id(self._workspace_id).team_id
        elif self._team_id is not None and self._workspace_id is not None:
            actual_workspace_team_id = self._api.workspace.get_info_by_id(
                self._workspace_id
            ).team_id
            if self._team_id != actual_workspace_team_id:
                raise ValueError(
                    f"Provided team_id {self._team_id} does not match the team of the workspace {self._workspace_id}."
                )
        self.team_selector.set_value(self._team_id)
        self.workspace_selector.set_value(self._workspace_id)
        self._style = (
            "justify-content: center; align-items: center;" if direction == "horizontal" else ""
        )
        self._selectors_style = selectors_style
        self._gap = 20 if direction == "horizontal" else 10
        self._changes_handled = False
        self._value_changed_callback = None
        super().__init__(widget_id=widget_id, file_path=__file__)

    @property
    def _content(self):
        return Container(
            [
                self.team_label,
                Container([self.team_selector], style=self._selectors_style),
                self.workspace_label,
                Container([self.workspace_selector], style=self._selectors_style),
            ],
            direction=self._direction,
            style=self._style,
            gap=self._gap,
        )

    @property
    def team_selector(self):
        if not hasattr(self, "_team_selector"):
            items = [Select.Item(team.id, team.name) for team in self._api.team.get_list()]
            select = Select(items=items, size=self._size)

            @select.value_changed
            def on_team_change(value: int):
                self._team_id = value
                self._update_workspace_selector()

            self._team_selector = select
        return self._team_selector

    @property
    def workspace_selector(self):
        if not hasattr(self, "_workspace_selector"):
            items = [
                Select.Item(ws.id, ws.name) for ws in self._api.workspace.get_list(self._team_id)
            ]
            select = Select(items=items, size=self._size)

            @select.value_changed
            def on_workspace_change(value: int):
                self._workspace_id = value
                StateJson()[self.widget_id]["workspaceId"] = value
                StateJson().send_changes()

                if self._changes_handled:
                    result = {"teamId": self._team_id, "workspaceId": value}
                    self._value_changed_callback(result)

            self._workspace_selector = select
        return self._workspace_selector

    @property
    def team_label(self):
        if not hasattr(self, "_team_label"):
            if self._show_label:
                self._team_label = Text(text="Team")
            else:
                self._team_label = Empty()
        return self._team_label

    @property
    def workspace_label(self):
        if not hasattr(self, "_workspace_label"):
            if self._show_label:
                self._workspace_label = Text(text="Workspace")
            else:
                self._workspace_label = Empty()
        return self._workspace_label

    def get_selected_team_id(self) -> Optional[int]:
        return StateJson()[self.widget_id]["teamId"]

    def get_selected_workspace_id(self) -> Optional[int]:
        return StateJson()[self.widget_id]["workspaceId"]

    def set_team_id(self, team_id: int):
        self._team_id = team_id
        self.team_selector.set_value(team_id)
        self._update_workspace_selector()

    def set_workspace_id(self, workspace_id: int):
        self._workspace_id = workspace_id
        self.workspace_selector.set_value(workspace_id)

    def _update_workspace_selector(self):
        items = [Select.Item(ws.id, ws.name) for ws in self._api.workspace.get_list(self._team_id)]
        self.workspace_selector.set(items)
        item_value_to_item = {item.value: item for item in items}
        current_item = item_value_to_item.get(self._workspace_id, None)
        if current_item is None:
            self._workspace_id = items[0].value
        self.workspace_selector.set_value(self._workspace_id)

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {"teamId": self._team_id, "workspaceId": self._workspace_id}

    def get_value(self):
        """Returns the current workspace selection info"""
        return {"teamId": self._team_id, "workspaceId": self._workspace_id}

    def value_changed(self, func):
        """Register a callback function to be called when team or workspace selection changes"""
        self._changes_handled = True
        self._value_changed_callback = func

        return func


class ProjectDatasetTable(Widget):
    class CurrentTable:
        PROJECTS = "projects"
        DATASETS = "datasets"

    def __init__(
        self,
        sort_by: Literal["name", "date", "assets"] = "date",
        sort_order: Literal["asc", "desc"] = "desc",
        page_size: int = 10,
        width: str = "auto",
        allowed_project_types: Optional[List[ProjectType]] = None,
        team_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        project_filter_fn: Optional[Callable] = None,
        widget_id: Optional[str] = None,
    ):
        self._api = Api()
        self._team_id = team_id or env.team_id()
        self._workspace_id = workspace_id or env.workspace_id()
        self._allowed_project_types = allowed_project_types
        if isinstance(self._allowed_project_types, list):
            if all(isinstance(pt, ProjectType) for pt in self._allowed_project_types):
                self._allowed_project_types = [pt.value for pt in self._allowed_project_types]
        self._project_id_to_info = {}
        self._datasets: List[DatasetInfo] = []
        self._dataset_id_to_info = {}

        self._id_to_search_str = {}
        self._id_to_sort_key = {}
        self._selected_project_id = None

        self._sort_by_key = sort_by
        self._sort_order = sort_order
        self._page_size = page_size
        self._width = width
        self._project_filter_fn = lambda p: True if project_filter_fn is None else project_filter_fn
        super().__init__(widget_id=widget_id, file_path=__file__)
        self._content = self.table

    @property
    def current_table(self):
        if not hasattr(self, "_current_table"):
            self._current_table = self.CurrentTable.PROJECTS
        return self._current_table

    @current_table.setter
    def current_table(self, value):
        if value not in self.CurrentTable.__dict__.values():
            raise ValueError(f"Invalid table type: {value}")
        self._current_table = value

    @property
    def table(self) -> FastTable:
        if not hasattr(self, "_table"):
            columns = self._get_columns()
            header_left_content = None
            if self.current_table == self.CurrentTable.PROJECTS:
                header_left_content = self.team_workspace_selector
            self._table = FastTable(
                data=self._get_table_data(),
                columns=columns,
                columns_options=self._get_column_options(len(columns)),
                page_size=self._page_size,
                sort_order=self._sort_order,
                sort_column_idx=self._get_sort_column_idx(),
                width=self._width,
                is_selectable=True,
                header_left_content=header_left_content,
                max_selected_rows=self._get_max_selected_rows(),
                search_position=self._get_search_position(),
            )
            self._table.set_search(self._search_function)
            self._table.set_sort(self._sort_function)
        return self._table

    @property
    def team_workspace_selector(self):
        if not hasattr(self, "_team_workspace_selector"):
            self._team_workspace_selector = TeamWorkspaceSelect(
                direction="horizontal",
                default_team_id=self._team_id,
                default_workspace_id=self._workspace_id,
                selectors_style="margin-left: -12px; border: 1px solid #779ab94a; border-radius: 6px; padding: 0px 12px; box-sizing: border-box;",
            )

            @self._team_workspace_selector.value_changed
            def on_value_change(value):
                self._team_id = value["teamId"]
                self._workspace_id = value["workspaceId"]
                self.current_table = self.CurrentTable.PROJECTS
                try:
                    self.table.loading = True
                    self._refresh_table()
                    self.table.clear_selection()
                finally:
                    self.table.loading = False

        return self._team_workspace_selector

    def set_project_filter(self, filter_fn: Callable):
        """Set a filter function to filter projects displayed in the table."""
        self._project_filter_fn = filter_fn
        if self.current_table == self.CurrentTable.PROJECTS:
            try:
                self.table.loading = True
                self._refresh_table()
                self.table.clear_selection()
            finally:
                self.table.loading = False

    def switch_table(self, table: Literal["projects", "datasets"]):
        if self.current_table == table:
            return

        if table == self.CurrentTable.PROJECTS:
            self._current_table = self.CurrentTable.PROJECTS
            self._selected_dataset_ids = None
            self.team_workspace_selector.show()
        elif table == self.CurrentTable.DATASETS:
            if not self._get_selected_project_id():
                return
            self._current_table = self.CurrentTable.DATASETS
            self._selected_project_id = self._get_selected_project_id()
            self.team_workspace_selector.hide()

        try:
            self.table.loading = True
            self._refresh_table()
        finally:
            self.table.loading = False

    def _refresh_table(self):
        from supervisely.app import DataJson, StateJson

        self.table._max_selected_rows = self._get_max_selected_rows()
        DataJson()[self.table.widget_id]["options"][
            "maxSelectedRows"
        ] = self.table._max_selected_rows

        self.table._search_position = self._get_search_position()
        DataJson()[self.table.widget_id]["options"]["searchPosition"] = self.table._search_position
        data = self._get_table_data()
        columns = self._get_columns()
        data_json = self.table.to_json()
        data_json["data"] = data
        item_ids = [row[1] for row in data]
        if item_ids != list(self._id_to_search_str.keys()):
            self._get_search_sort_keys()
            assert item_ids == list(self._id_to_search_str.keys())
        self.table.read_json(data_json, custom_columns=columns)
        self.table._refresh()

    def _get_selected_project_id(self) -> Optional[int]:
        selected_row = self.table.get_selected_row()
        if selected_row is None:
            return None
        return selected_row.row[1]

    def _get_selected_project(self) -> Optional[ProjectInfo]:
        return self._project_id_to_info.get(self._get_selected_project_id(), None)

    def get_selected_project_id(self) -> Optional[int]:
        return self._selected_project_id

    def get_selected_project(self) -> Optional[ProjectInfo]:
        return self._project_id_to_info.get(self._selected_project_id, None)

    def _get_selected_dataset_ids(self) -> Optional[List[int]]:
        if not self.current_table == self.CurrentTable.DATASETS:
            return None
        rows = []
        for row in self.table.get_selected_rows():
            if row is None:
                continue
            rows.append(row.row[1])
        return rows

    def get_selected_dataset_ids(self) -> Optional[List[int]]:
        return self._get_selected_dataset_ids()

    def get_selected_datasets(self) -> Optional[List[DatasetInfo]]:
        dataset_ids = self._get_selected_dataset_ids()
        return [self._dataset_id_to_info.get(did, None) for did in dataset_ids]

    def get_selected_datasets_full_names(self) -> Optional[List[str]]:
        dataset_ids = self._get_selected_dataset_ids()
        return [self._dataset_id_to_full_name.get(did, None) for did in dataset_ids]

    def has_nested_datasets(self) -> bool:
        return any("/" in full_name for full_name in self._dataset_id_to_full_name.values())

    def set_team(self, team_id: int):
        if not self.current_table == self.CurrentTable.PROJECTS:
            return
        self.team_workspace_selector.set_team_id(team_id)
        self._refresh_table()

    def set_workspace(self, workspace_id: int):
        if not self.current_table == self.CurrentTable.PROJECTS:
            return
        self.team_workspace_selector.set_workspace_id(workspace_id)
        self._refresh_table()

    def get_json_data(self):
        return {
            "selectedProject": self._get_selected_project_id(),
            "selectedDatasets": self._get_selected_dataset_ids(),
        }

    def get_json_state(self):
        return {"currentTable": self.current_table}

    def hide(self):
        self.table.hide()

    def show(self):
        self.table.show()

    def disable(self):
        self.table.disable()

    def enable(self):
        self.table.enable()

    def _sort_function(self, data: pd.DataFrame, column_idx: int, order: str = "asc"):
        if not self._id_to_sort_key:
            return data
        data = data.copy()
        first_sort_keys = next(iter(self._id_to_sort_key.values()), [])
        if column_idx >= len(first_sort_keys) if self._id_to_sort_key else True:
            raise IndexError(
                f"Sorting by column idx = {column_idx} is not possible, your sort values have only {len(first_sort_keys) if self._id_to_sort_key else 0} columns with idx from 0 to {len(first_sort_keys) - 1 if self._id_to_sort_key else -1}"
            )

        if order == "asc":
            ascending = True
        else:
            ascending = False

        try:
            current_sort_keys = []
            for row in data.values:
                p_id = row[1]
                sort_key = self._id_to_sort_key.get(p_id, [])[column_idx]
                current_sort_keys.append(sort_key)

            sort_series = pd.Series(current_sort_keys, index=data.index)
            sorted_indices = sort_series.sort_values(ascending=ascending).index
            data = data.loc[sorted_indices]
            data.reset_index(inplace=True, drop=True)

        except IndexError as e:
            e.args = (
                f"Sorting by column idx = {column_idx} is not possible, your sort values have only {len(first_sort_keys) if self._id_to_sort_key else 0} columns with idx from 0 to {len(first_sort_keys) - 1 if self._id_to_sort_key else -1}",
            )
            raise e

        return data

    def _search_function(self, data: pd.DataFrame, search_value: str):
        search_texts = []
        for row in data.values:
            p_id = row[1]
            search_key = self._id_to_search_str.get(p_id, "")
            search_texts.append(search_key)
        search_series = pd.Series(search_texts, index=data.index)
        mask = search_series.str.contains(search_value, case=False, na=False)
        return data[mask]

    def _get_table_data(self):
        if self.current_table == self.CurrentTable.PROJECTS:
            workspace_id = self.team_workspace_selector.get_selected_workspace_id()
            return self._get_projects_data(workspace_id=workspace_id)
        elif self.current_table == self.CurrentTable.DATASETS:
            return self._get_datasets_data(project_id=self.get_selected_project_id())
        self._get_search_sort_keys()

    def _get_projects_data(self, workspace_id):
        if not workspace_id:
            self._project_id_to_info = {}
            self._id_to_search_str = {}
            self._id_to_sort_key = {}
            return []
        filters = None
        if self._allowed_project_types:
            filters = [
                {
                    ApiField.FIELD: ApiField.TYPE,
                    ApiField.OPERATOR: "in",
                    ApiField.VALUE: self._allowed_project_types,
                }
            ]

        projects = self._api.project.get_list(
            workspace_id,
            fields=[
                ApiField.ID,
                ApiField.NAME,
                ApiField.TYPE,
                ApiField.CREATED_AT,
                ApiField.IMAGES_COUNT,
                ApiField.REFERENCE_IMAGE_URL,
            ],
            filters=filters,
        )

        id_to_info = {p.id: p for p in list(filter(self._project_filter_fn, projects))}
        self._project_id_to_info = id_to_info

        table_data = []
        for p_id, info in self._project_id_to_info.items():
            project_thumbnail = ProjectThumbnail(info, remove_margins=True)
            ds_thumb = DatasetThumbnail(show_project_name=False, remove_margins=True)
            ds_thumb.hide()
            dt = datetime.strptime(info.created_at.replace("Z", ""), "%Y-%m-%dT%H:%M:%S.%f")
            row_data = [
                self._widget_to_cell_value(Container([project_thumbnail, ds_thumb], gap=0)),
                p_id,
                dt.strftime("%d %b %Y %H:%M"),
                info.type.replace("_", " ").title(),
                info.items_count or 0,
            ]
            table_data.append(row_data)

        return table_data

    def _get_search_sort_keys(self):
        self._id_to_search_str = {}
        self._id_to_sort_key = {}
        mapping = (
            self._project_id_to_info
            if self.current_table == self.CurrentTable.PROJECTS
            else self._dataset_id_to_info
        )
        for p_id, info in mapping.items():
            dt = datetime.strptime(info.created_at.replace("Z", ""), "%Y-%m-%dT%H:%M:%S.%f")
            search_value = [
                info.name.lower(),
                p_id,
                dt.timestamp(),
                info.items_count,
            ]
            if self.current_table == self.CurrentTable.PROJECTS:
                search_value.insert(3, info.type.replace("_", " ").lower())
            self._id_to_search_str[p_id] = "".join(str(v) for v in search_value)
            self._id_to_sort_key[p_id] = search_value

    def _get_datasets_data(self, project_id):
        project_info = self._project_id_to_info.get(project_id, None)
        if project_info is None:
            return []

        datasets: List[DatasetInfo] = []
        id_to_full_name = {}
        for parents, dataset_info in self._api.dataset.tree(project_id):
            full_name = "/".join(parents + [dataset_info.name])
            id_to_full_name[dataset_info.id] = full_name
            datasets.append(dataset_info)

        self._dataset_id_to_info = {d.id: d for d in datasets}
        self._dataset_id_to_full_name = id_to_full_name

        table_data = []
        for dataset_id, dataset in self._dataset_id_to_info.items():
            proj_thumb = ProjectThumbnail(remove_margins=True)
            proj_thumb.hide()
            full_name = id_to_full_name.get(dataset.id, dataset.name)
            ds_preview = DatasetThumbnail(
                project_info,
                dataset,
                show_project_name=False,
                remove_margins=True,
                custom_name=full_name,
            )
            dt = datetime.strptime(dataset.created_at.replace("Z", ""), "%Y-%m-%dT%H:%M:%S.%f")
            row_data = [
                self._widget_to_cell_value(Container([proj_thumb, ds_preview], gap=0)),
                dataset_id,
                dt.strftime("%d %b %Y %H:%M"),
                dataset.items_count or 0,
            ]
            table_data.append(row_data)

        return table_data

    def _get_columns(self):
        if self.current_table == self.CurrentTable.PROJECTS:
            return [
                (
                    "Name",
                    Container(
                        [
                            ProjectThumbnail(remove_margins=True),
                            DatasetThumbnail(show_project_name=False, remove_margins=True),
                        ],
                        gap=0,
                    ),
                ),
                "ID",
                "Date Modified",
                "Type",
                "Assets",
            ]
        elif self.current_table == self.CurrentTable.DATASETS:
            return [
                (
                    "Name",
                    Container(
                        [
                            ProjectThumbnail(remove_margins=True),
                            DatasetThumbnail(show_project_name=False, remove_margins=True),
                        ],
                        gap=0,
                    ),
                ),
                "ID",
                "Created At",
                "Items",
            ]

    def _get_max_selected_rows(self):
        if self.current_table == self.CurrentTable.PROJECTS:
            return 1
        elif self.current_table == self.CurrentTable.DATASETS:
            return None

    def _get_search_position(self):
        if self.current_table == self.CurrentTable.PROJECTS:
            return "right"
        elif self.current_table == self.CurrentTable.DATASETS:
            return "left"

    def _get_sort_column_idx(self):
        if self.current_table == self.CurrentTable.PROJECTS:
            return {"date": 2, "assets": 4}.get(self._sort_by_key, 2)
        elif self.current_table == self.CurrentTable.DATASETS:
            return {"date": 2, "assets": 3}.get(self._sort_by_key, 2)

    def _get_column_options(self, col_cnt):
        return [{"customCell": True}] + [{}] * (col_cnt - 1)

    def _widget_to_cell_value(self, widget: Widget) -> str:
        if isinstance(widget, Container):
            return json.dumps(
                {
                    "widget_id": widget.widget_id,
                    "widgets": [w.widget_id for w in widget._widgets],
                }
            )
        else:
            return json.dumps({"widget_id": widget.widget_id, "widgets": []})
