from typing import Callable, Dict, List, Optional, Union

import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app.widgets import Widget
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.select.select import Select
from supervisely.app.widgets.tree_select.tree_select import TreeSelect
from supervisely.project.project_type import ProjectType


class SelectCollection(Widget):
    """SelectCollection widget in Supervisely is a widget that allows users to select one or multiple collections (entities collections).
    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/selection/selectCollection>`_
        (including screenshots and examples).

    :param default_id: The ID of the collection to be selected by default.
    :type default_id: Union[int, None]
    :param project_id: The ID of the project to read collections from.
    :type project_id: Union[int, None]
    :param multiselect: Whether multiple collections can be selected.
    :type multiselect: bool
    :param compact: Whether the widget should be compact (e.g. no team, workspace, and project selectors).
    :type compact: bool
    :param select_all_collections: Whether all collections should be selected by default.
    :type select_all_collections: bool
    :param allowed_project_types: The list of project types that are allowed to be selected.
    :type allowed_project_types: Optional[List[ProjectType]]
    :param team_is_selectable: Whether the team selector should be selectable.
    :type team_is_selectable: bool
    :param workspace_is_selectable: Whether the workspace selector should be selectable.
    :type workspace_is_selectable: bool
    :param widget_id: The unique identifier of the widget.
    :type widget_id: Union[str, None]
    :param show_select_all_collections_checkbox: Whether the checkbox to select all collections should be shown.
    :type show_select_all_collections_checkbox: bool

    :Public methods:
    - `set_project_id(project_id: int) -> None`: Set the project ID to read collections from.
    - `get_selected_ids() -> Optional[List[int]]`: Get the IDs of the selected collections.
    - `get_selected_id() -> Optional[int]`: Get the ID of the selected collection.
    - `value_changed(func: Callable) -> Callable`: Decorator to set the callback function for the value changed event.
    - `set_collection_id(collection_id: int) -> None`: Set the ID of the collection to be selected by default.
    - `set_collection_ids(collection_ids: List[int]) -> None`: Set the IDs of the collections to be selected by default.
    - `get_selected_project_id() -> Optional[int]`: Get the ID of the selected project.
    - `get_selected_team_id() -> int`: Get the ID of the selected team.
    - `set_team_id(team_id: int) -> None`: Set the team ID to read workspaces from.
    - `get_selected_workspace_id() -> int`: Get the ID of the selected workspace.
    - `set_workspace_id(workspace_id: int) -> None`: Set the workspace ID to read projects from.
    - `is_all_selected() -> bool`: Check if all collections are selected.
    - `select_all() -> None`: Select all collections.
    - `disable() -> None`: Disable the widget in the UI.
    - `enable() -> None`: Enable the widget in the UI.
    - `set_selected_id(collection_id: int) -> None`: Set the ID of the collection to be selected by default.
    - `set_selected_ids(collection_ids: List[int]) -> None`: Set the IDs of the collections to be selected by default.

    :Properties:
    - `team_id`: The ID of the team selected in the widget.
    - `workspace_id`: The ID of the workspace selected in the widget.
    - `project_id`: The ID of the project selected in the widget.

    :Usage example:

        .. code-block:: python
            from supervisely.app.widgets import SelectCollection

            project_id = 123
            collection_id = 456

            select_collection = SelectCollection(
                default_id=collection_id,
                project_id=project_id,
                multiselect=True,
            )

            @select_collection.value_changed
            def on_change(selected_ids):
                print(selected_ids) # Output: [456, 789]
    """

    def __init__(
        self,
        default_id: Union[int, None] = None,
        project_id: Union[int, None] = None,
        multiselect: bool = False,
        compact: bool = False,
        select_all_collections: bool = False,
        allowed_project_types: Optional[List[ProjectType]] = None,
        team_is_selectable: bool = True,
        workspace_is_selectable: bool = True,
        show_select_all_collections_checkbox: bool = True,
        widget_id: Union[str, None] = None,
        width: int = 193,
    ):
        self._api = Api.from_env()

        if default_id is not None and project_id is None:
            raise ValueError("Project ID must be provided when default collection ID is set.")

        if not multiselect and select_all_collections:
            raise ValueError("Select all collections is only available in multiselect mode.")

        # Reading team_id and workspace_id from environment variables.
        # If not found, error will be raised.
        self._team_id = env.team_id()
        self._workspace_id = env.workspace_id()

        # Using environment variables to set the default values if they are not provided.
        self._project_id = project_id or env.project_id(raise_not_found=False)
        self._collection_id = default_id

        # Get mapping of collection ID to name for current project.
        self._collections_names_map = None
        self._collections_ids_map = None

        self._multiselect = multiselect
        self._compact = compact

        # Extract values from Enum to match the .type property of the ProjectInfo object.

        self._project_types = None
        if allowed_project_types is not None:
            if all(allowed_project_types) is isinstance(allowed_project_types, ProjectType):
                self._project_types = (
                    [project_type.value for project_type in allowed_project_types]
                    if allowed_project_types is not None
                    else None
                )
            elif all(allowed_project_types) is isinstance(allowed_project_types, str):
                self._project_types = allowed_project_types

        # Widget components.
        self._select_team = None
        self._select_workspace = None
        self._select_project = None
        self._select_collection = None
        self._select_all_collections_checkbox = None
        self._width = width

        # List of widgets will be used to create a Container.
        self._widgets = []

        if not compact:
            # If the widget is not compact, create team, workspace, and project selectors.
            self._create_selectors(team_is_selectable, workspace_is_selectable)

        # Create the collection selector.
        self._create_collection_selector(select_all_collections)

        # Create the checkbox to select all collections if needed.
        if show_select_all_collections_checkbox:
            self._create_select_all_collections_checkbox(select_all_collections)

        # Group the selectors and the collection selector into a container.
        self._content = Container(self._widgets)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def disable(self):
        """Disable the widget in the UI."""
        for widget in self._widgets:
            widget.disable()

    def enable(self) -> None:
        """Enable the widget in the UI."""
        for widget in self._widgets:
            widget.enable()

    @property
    def team_id(self) -> int:
        """The ID of the team selected in the widget.

        :return: The ID of the team.
        :rtype: int
        """
        return self._team_id

    @team_id.setter
    def team_id(self, team_id: int) -> None:
        """Set the team ID to read workspaces from.

        :param team_id: The ID of the team.
        :type team_id: int
        """
        if not self._compact:
            self._select_team.set_value(team_id)
            self._select_workspace.set(self._get_select_items(team_id=team_id))
        self._team_id = team_id

    def get_selected_team_id(self) -> int:
        """Get the ID of the selected team.

        :return: The ID of the selected team.
        :rtype: int
        """
        return self.team_id

    def set_team_id(self, team_id: int) -> None:
        """Set the team ID to read workspaces from.

        :param team_id: The ID of the team.
        :type team_id: int
        """
        self.team_id = team_id

    @property
    def workspace_id(self) -> int:
        """The ID of the workspace selected in the widget.

        :return: The ID of the workspace.
        :rtype: int
        """
        return self._workspace_id

    @workspace_id.setter
    def workspace_id(self, workspace_id: int) -> None:
        """Set the workspace ID to read projects from.

        :param workspace_id: The ID of the workspace.
        :type workspace_id: int
        """
        if not self._compact:
            self._select_workspace.set_value(workspace_id)
            self._select_project.set(self._get_select_items(workspace_id=workspace_id))
        self._workspace_id = workspace_id

    def get_selected_workspace_id(self) -> int:
        """Get the ID of the selected workspace.

        :return: The ID of the selected workspace.
        :rtype: int
        """
        return self.workspace_id

    def set_workspace_id(self, workspace_id: int) -> None:
        """Set the workspace ID to read projects from.

        :param workspace_id: The ID of the workspace.
        :type workspace_id: int
        """
        self.workspace_id = workspace_id

    @property
    def project_id(self) -> Optional[int]:
        """The ID of the project selected in the widget.

        :return: The ID of the project.
        :rtype: Optional[int]
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id: Optional[int]) -> None:
        """Set the project ID to read collections from.

        :param project_id: The ID of the project.
        :type project_id: int
        """
        if not self._compact:
            self._select_project.set_value(project_id)
        self.set_project_id(project_id)

    def get_selected_project_id(self) -> Optional[int]:
        """Get the ID of the selected project.

        :return: The ID of the selected project.
        :rtype: Optional[int]
        """
        return self.project_id

    def set_collection(self, collection: Union[int, str]) -> None:
        """Set the collection to be selected.

        :param collection: The ID or name of the collection.
        :type collection: The ID or name of the collection.
        :raise ValueError: If multiselect is enabled.
        """
        if isinstance(collection, int):
            self.set_selected_id(collection)
        elif isinstance(collection, str):
            self.set_selected_name(collection)
        else:
            raise ValueError("Collection ID must be an integer or a string.")

    def set_collections(self, collections: Union[List[int], List[str]]) -> None:
        """Set the collections to be selected.

        :param collection: The ID or name of the collection.
        :type collection: The ID or name of the collection.
        :raise ValueError: If multiselect is disabled.
        """
        if not isinstance(collections, list):
            raise ValueError("Collections must be a list of integers or strings.")
        if all(isinstance(i, int) for i in collections) or len(collections) == 0:
            self.set_selected_ids(collections)
        elif all(isinstance(i, str) for i in collections):
            self.set_selected_names(collections)
        else:
            raise ValueError("Collection IDs must be a list of integers or a list of strings.")

    def value_changed(self, func: Callable) -> Callable:
        """Decorator to set the callback function for the value changed event.

        :param func: The callback function.
        :type func: Callable
        :return: The callback function.
        :rtype: Callable
        """

        @self._select_collection.value_changed
        def _click(items: Union[List[str], str]):
            if isinstance(items, list):
                res = [self._collections_names_map[item].id for item in items]
            else:
                res = self._collections_names_map[items].id

            func(res)

        return _click

    def _create_select_all_collections_checkbox(self, select_all_collections: bool) -> None:
        """Create the checkbox to select all collections.

        :param select_all_collections: Whether all collections should be selected by default.
        :type select_all_collections: bool
        """
        if not self._multiselect:
            # We'll only create the checkbox if multiselect is enabled.
            return
        select_all_collections_checkbox = Checkbox("Select all collections")

        @select_all_collections_checkbox.value_changed
        def select_all_collections_checkbox_handler(checked: bool) -> None:
            """Handler function for the event when the checkbox value changes.

            :param checked: The value of the checkbox.
            :type checked: bool
            """
            if self._project_id is None:
                return

            if checked:
                self.select_all()
            else:
                self.deselect_all()

        self._widgets.append(select_all_collections_checkbox)
        self._select_all_collections_checkbox = select_all_collections_checkbox
        if select_all_collections:
            self.select_all()

    def _create_collection_selector(self, select_all_collections: bool) -> None:
        """Create the collection selector.

        :param select_all_collections: Whether all collections should be selected by default.
        :type select_all_collections: bool
        """
        items = self._read_collections(self._project_id)
        if items is None or len(items) == 0:
            items = []
        self._select_collection = Select(
            items=items,
            multiple=self._multiselect,
            width_px=self._width,
            placeholder="Select collection",
            filterable=True,
        )
        if self._collection_id is not None:
            info = self._collections_ids_map.get(self._collection_id)
            if info is not None:
                value = [info.name] if self._multiselect else info.name
                self._select_collection.set_value(value)
        if select_all_collections:
            self.select_all()

        # Adding the collection selector to the list of widgets to be added to the container.
        self._widgets.append(self._select_collection)

    def _create_selectors(self, team_is_selectable: bool, workspace_is_selectable: bool):
        """Create the team, workspace, and project selectors.

        :param team_is_selectable: Whether the team selector should be selectable.
        :type team_is_selectable: bool
        :param workspace_is_selectable: Whether the workspace selector should be selectable.
        :type workspace_is_selectable: bool
        """

        def team_selector_handler(team_id: int):
            """Handler function for the event when the team selector value changes.

            :param team_id: The ID of the selected team.
            :type team_id: int
            """
            self._select_workspace.set(items=self._get_select_items(team_id=team_id))
            self._team_id = team_id

        def workspace_selector_handler(workspace_id: int):
            """Handler function for the event when the workspace selector value changes.

            :param workspace_id: The ID of the selected workspace.
            :type workspace_id: int
            """
            self._select_project.set(items=self._get_select_items(workspace_id=workspace_id))
            self._workspace_id = workspace_id

        def project_selector_handler(project_id: int):
            """Handler function for the event when the project selector value changes.

            :param project_id: The ID of the selected project.
            :type project_id: int
            """
            self._select_collection.set(self._read_collections(project_id))
            self._project_id = project_id

            if (
                self._select_all_collections_checkbox is not None
                and self._select_all_collections_checkbox.is_checked()
            ):
                self.select_all()
                self._select_collection.hide()

        self._select_team = Select(
            items=self._get_select_items(),
            placeholder="Select team",
            filterable=True,
            width_px=self._width,
        )
        self._select_team.set_value(self._team_id)
        if not team_is_selectable:
            self._select_team.disable()

        self._select_workspace = Select(
            items=self._get_select_items(team_id=self._team_id),
            placeholder="Select workspace",
            filterable=True,
            width_px=self._width,
        )
        self._select_workspace.set_value(self._workspace_id)
        if not workspace_is_selectable:
            self._select_workspace.disable()

        self._select_project = Select(
            items=self._get_select_items(workspace_id=self._workspace_id),
            placeholder="Select project",
            filterable=True,
            width_px=self._width,
        )
        self._select_project.set_value(self._project_id)

        # Register the event handlers.
        self._select_team.value_changed(team_selector_handler)
        self._select_workspace.value_changed(workspace_selector_handler)
        self._select_project.value_changed(project_selector_handler)

        # Adding widgets to the list, so they can be added to the container.
        self._widgets.extend([self._select_team, self._select_workspace, self._select_project])

    def _get_select_items(self, **kwargs) -> List[Select.Item]:
        """Get the list of items for the team, workspace, and project selectors.
        Possible keyword arguments are 'team_id' and 'workspace_id'.

        :return: The list of items.
        :rtype: List[Select.Item]
        """
        if not kwargs:
            items = self._api.team.get_list()
        elif "team_id" in kwargs:
            items = self._api.workspace.get_list(kwargs["team_id"])
        elif "workspace_id" in kwargs:
            projects_list = self._api.project.get_list(kwargs["workspace_id"])
            if self._project_types is not None:
                items = [
                    project for project in projects_list if project.type in self._project_types
                ]  # TODO: Filter project from API, not from here.
            else:
                items = projects_list

        return [Select.Item(value=item.id, label=item.name) for item in items]

    def get_json_data(self) -> Dict:
        """Get the JSON data of the widget.

        :return: The JSON data.
        :rtype: Dict
        """
        return {}

    def get_json_state(self) -> Dict:
        """Get the JSON state of the widget.

        :return: The JSON state.
        :rtype: Dict
        """
        return {}

    def _read_collections(self, project_id: Optional[int]) -> Optional[List[Select.Item]]:
        """Get the lisf of Select.Item objects representing the collection hierarchy.

        :param project_id: The ID of the project.
        :type project_id: Optional[int]
        :return: The list of Select.Item objects.
        :rtype: Optional[List[Select.Item]]
        """
        self._fetch_collections(project_id)

        if not self._collections_names_map or not project_id:
            return None

        collections = [Select.Item(i.name, i.name) for i in self._collections_names_map.values()]
        return collections

    def _fetch_collections(self, project_id: Optional[int]) -> None:
        """Get the mapping of collection name to EntitiesCollectionInfo object.

        :param project_id: The ID of the project.
        :type project_id: Optional[int]
        :return: None
        :rtype: None
        """
        self._collections_names_map = {}
        self._collections_ids_map = {}

        if not project_id:
            return

        for collection in self._api.entities_collection.get_list(project_id):
            self._collections_names_map[collection.name] = collection
            self._collections_ids_map[collection.id] = collection

    def set_project_id(self, project_id: Optional[int]) -> None:
        """Set the project ID to read collections from.

        :param project_id: The ID of the project.
        :type project_id: int
        """
        self._project_id = project_id
        items = self._read_collections(project_id)
        if items is None or len(items) == 0:
            self._select_collection.set([])
            return
        self._select_collection.set(items)
        if self._multiselect:
            if self._select_all_collections_checkbox.is_checked():
                self.select_all()
            else:
                self.deselect_all()
        else:
            self._select_collection.set_value("")

    def _get_selected(self) -> Optional[Union[List[int], int]]:
        """Get the ID of the selected collection(s).

        :return: The ID of the selected collection(s).
        :rtype: Optional[Union[List[int], int]]
        """
        selected = self._select_collection.get_value()
        if not selected:
            return None

        if isinstance(selected, list):
            return [self._collections_names_map[item].id for item in selected]
        else:
            return self._collections_names_map[selected].id

    def get_selected_ids(self) -> Optional[List[int]]:
        """Get the IDs of the selected collections.

        raise ValueError if multiselect is disabled.
        return: The IDs of the selected collections.
        rtype: Optional[List[int]]
        """
        if not self._multiselect:
            raise ValueError("This method can only be called when multiselect is enabled.")
        return self._get_selected()

    def get_selected_id(self) -> Optional[int]:
        """Get the ID of the selected collection.

        raise ValueError if multiselect is enabled.
        return: The ID of the selected collection.
        rtype: Optional[int]
        """
        if self._multiselect:
            raise ValueError("This method can only be called when multiselect is disabled.")
        return self._get_selected()

    def set_selected_ids(self, collection_ids: List[int]) -> None:
        """Set the IDs of the collections to be selected by default.

        :param collection_ids: The IDs of the collections.
        :type collection_ids: List[int]
        """
        if not self._multiselect:
            raise ValueError("This method can only be called when multiselect is enabled.")
        self._set_selected_by_ids(collection_ids)

    def set_selected_id(self, collection_id: int) -> None:
        """Set the ID of the collection to be selected by default.

        :param collection_id: The ID of the collection.
        :type collection_id: int
        """
        if self._multiselect:
            raise ValueError("This method can only be called when multiselect is disabled.")
        self._set_selected_by_ids([collection_id])

    def _set_selected_by_ids(self, collection_ids: List[int]) -> None:
        """Set the IDs of the collections to be selected by default.

        :param collection_ids: The IDs of the collections.
        :type collection_ids: List[int]
        """
        if not self._collections_ids_map:
            return

        selected = []
        for i in collection_ids:
            if i in self._collections_ids_map:
                selected.append(self._collections_ids_map[i].name)

        if self._multiselect:
            if len(selected) == len(self._collections_ids_map):
                self.select_all()
            else:
                self.deselect_all()
                self._select_collection.set_value(selected)

        else:
            if len(selected) > 1:
                raise ValueError("More than one collection found, but multiselect is disabled.")
            value = selected[0] if selected else ""
            self._select_collection.set_value(value)

    def set_selected_name(self, collection_name: str) -> None:
        """Set the name of the collection to be selected by default.

        :param collection_name: The name of the collection.
        :type collection_name: str
        """
        if self._multiselect:
            raise ValueError("This method can only be called when multiselect is disabled.")
        self._set_selected_by_names([collection_name])

    def set_selected_names(self, collection_names: List[str]) -> None:
        """Set the names of the collections to be selected by default.

        :param collection_names: The names of the collections.
        :type collection_names: List[str]
        """
        if not self._multiselect:
            raise ValueError("This method can only be called when multiselect is enabled.")
        self._set_selected_by_names(collection_names)

    def _set_selected_by_names(self, collection_names: List[str]) -> None:
        """Set the names of the collections to be selected by default.

        :param collection_names: The names of the collections.
        :type collection_names: List[str]
        """
        if not self._collections_names_map:
            return

        ids = []
        for i in collection_names:
            if i in self._collections_names_map:
                ids.append(self._collections_names_map[i].id)
        self._set_selected_by_ids(ids)

    def is_all_selected(self) -> bool:
        """Check if all collections are selected.

        return: True if all collections are selected, False otherwise.
        rtype: bool
        """
        return len(self._select_collection.get_value()) == len(self._collections_names_map)

    def select_all(self) -> None:
        """Select all collections."""
        if not self._multiselect:
            raise ValueError("This method can only be called when multiselect is enabled.")
        self._select_collection.set_value(list(self._collections_names_map.keys()))
        if self._select_all_collections_checkbox is not None:
            self._select_all_collections_checkbox.check()
            self._select_collection.hide()

    def deselect_all(self) -> None:
        """Deselect all collections."""
        if not self._multiselect:
            raise ValueError("This method can only be called when multiselect is enabled.")
        self._select_collection.set_value([])
        if self._select_all_collections_checkbox is not None:
            self._select_all_collections_checkbox.uncheck()
            self._select_collection.show()
