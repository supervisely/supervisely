from typing import Callable, Dict, List, Optional, Union

import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app.widgets import Widget
from supervisely.app.widgets.checkbox.checkbox import Checkbox
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.field.field import Field
from supervisely.app.widgets.select.select import Select
from supervisely.app.widgets.tree_select.tree_select import TreeSelect
from supervisely.project.project_type import ProjectType


class SelectDatasetTree(Widget):
    """SelectDatasetTree widget in Supervisely is a widget that allows users to select datasets from a tree-like structure.
    Read about it in `Developer Portal <https://developer.supervisely.com/app-development/widgets/selection/selectdatasettree>`_
        (including screenshots and examples).

    :param default_id: The ID of the dataset to be selected by default.
    :type default_id: Union[int, None]
    :param project_id: The ID of the project to read datasets from.
    :type project_id: Union[int, None]
    :param multiselect: Whether multiple datasets can be selected.
    :type multiselect: bool
    :param compact: Whether the widget should be compact (e.g. no team, workspace, and project selectors).
    :type compact: bool
    :param select_all_datasets: Whether all datasets should be selected by default.
    :type select_all_datasets: bool
    :param allowed_project_types: The list of project types that are allowed to be selected.
    :type allowed_project_types: Optional[List[ProjectType]]
    :param flat: Whether the dataset selector should be flat.
    :type flat: bool
    :param always_open: Whether the dataset selector should always be open.
    :type always_open: bool
    :param team_is_selectable: Whether the team selector should be selectable.
    :type team_is_selectable: bool
    :param workspace_is_selectable: Whether the workspace selector should be selectable.
    :type workspace_is_selectable: bool
    :param append_to_body: Determines where the popover is attached. If False, it is positioned inside the input's container. This can cause the popover to be hidden if the input is within a Card or a widget that restricts visibility.
    :type append_to_body: bool
    :param widget_id: The unique identifier of the widget.
    :type widget_id: Union[str, None]
    :param show_select_all_datasets_checkbox: Whether the checkbox to select all datasets should be shown.
    :type show_select_all_datasets_checkbox: bool

    :Public methods:
    - `set_project_id(project_id: int) -> None`: Set the project ID to read datasets from.
    - `get_selected_ids() -> Optional[List[int]]`: Get the IDs of the selected datasets.
    - `get_selected_id() -> Optional[int]`: Get the ID of the selected dataset.
    - `value_changed(func: Callable) -> Callable`: Decorator to set the callback function for the value changed event.
    - `set_dataset_id(dataset_id: int) -> None`: Set the ID of the dataset to be selected by default.
    - `set_dataset_ids(dataset_ids: List[int]) -> None`: Set the IDs of the datasets to be selected by default.
    - `get_selected_project_id() -> Optional[int]`: Get the ID of the selected project.
    - `get_selected_team_id() -> int`: Get the ID of the selected team.
    - `set_team_id(team_id: int) -> None`: Set the team ID to read workspaces from.
    - `get_selected_workspace_id() -> int`: Get the ID of the selected workspace.
    - `set_workspace_id(workspace_id: int) -> None`: Set the workspace ID to read projects from.
    - `is_all_selected() -> bool`: Check if all datasets are selected.
    - `select_all() -> None`: Select all datasets.

    :Properties:
    - `team_id`: The ID of the team selected in the widget.
    - `workspace_id`: The ID of the workspace selected in the widget.
    - `project_id`: The ID of the project selected in the widget.

    :Usage example:

        .. code-block:: python
            from supervisely.app.widgets import SelectDatasetTree

            project_id = 123
            dataset_id = 456

            select_dataset_tree = SelectDatasetTree(
                default_id=dataset_id,
                project_id=project_id,
                multiselect=True,
                flat=True)

            @select_dataset_tree.value_changed
            def on_change(selected_ids):
                print(selected_ids) # Output: [456, 789]
    """

    def __init__(
        self,
        default_id: Union[int, None] = None,
        project_id: Union[int, None] = None,
        multiselect: bool = False,
        compact: bool = False,
        select_all_datasets: bool = False,
        allowed_project_types: Optional[List[ProjectType]] = None,
        flat: bool = False,
        always_open: bool = False,
        team_is_selectable: bool = True,
        workspace_is_selectable: bool = True,
        append_to_body: bool = True,
        widget_id: Union[str, None] = None,
        show_select_all_datasets_checkbox: bool = True,
        width: int = 193,
        show_selectors_labels: bool = False,
    ):
        self._api = Api.from_env()

        if default_id is not None and project_id is None:
            raise ValueError("Project ID must be provided when default dataset ID is set.")

        if not multiselect and select_all_datasets:
            raise ValueError("Select all datasets is only available in multiselect mode.")

        # Reading team_id and workspace_id from environment variables.
        # If not found, error will be raised.
        self._team_id = env.team_id()
        self._workspace_id = env.workspace_id()

        # Using environment variables to set the default values if they are not provided.
        self._project_id = project_id or env.project_id(raise_not_found=False)
        self._dataset_id = default_id or env.dataset_id(raise_not_found=False)
        if self._project_id:
            project_info = self._api.project.get_info_by_id(self._project_id)
            if allowed_project_types is not None:
                allowed_values = []
                if not isinstance(allowed_project_types, list):
                    allowed_project_types = [allowed_project_types]

                for pt in allowed_project_types:
                    if isinstance(pt, (ProjectType, str)):
                        allowed_values.append(str(pt))

                if project_info.type not in allowed_values:
                    self._project_id = None

        self._multiselect = multiselect
        self._compact = compact
        self._append_to_body = append_to_body

        # User-defined callbacks
        self._team_changed_callbacks = []
        self._workspace_changed_callbacks = []
        self._project_changed_callbacks = []

        # Extract values from Enum to match the .type property of the ProjectInfo object.
        self._project_types = None
        if allowed_project_types is not None:
            self._project_types = []
            for project_type in allowed_project_types:
                if isinstance(project_type, ProjectType):
                    project_type = project_type.value
                elif not isinstance(project_type, str):
                    continue

                self._project_types.append(project_type)

            if self._project_types == []:
                self._project_types = None

        # Widget components.
        self._select_team = None
        self._select_workspace = None
        self._select_project = None
        self._select_dataset = None
        self._width = width

        # Flags
        self._team_is_selectable = team_is_selectable
        self._workspace_is_selectable = workspace_is_selectable

        # List of widgets will be used to create a Container.
        self._widgets = []

        if not compact:
            # If the widget is not compact, create team, workspace, and project selectors.
            self._create_selectors(team_is_selectable, workspace_is_selectable)

        # Create the dataset selector.
        self._create_dataset_selector(flat, always_open, select_all_datasets)

        # Create the checkbox to select all datasets if needed.
        self._select_all_datasets_checkbox = None
        if show_select_all_datasets_checkbox:
            self._create_select_all_datasets_checkbox(select_all_datasets)

        self._show_selectors_labels = show_selectors_labels
        # Group the selectors and the dataset selector into a container.
        self._content = Container(self._widgets)
        super().__init__(widget_id=widget_id, file_path=__file__)

    def disable(self):
        """Disable the widget in the UI."""
        for widget in self._widgets:
            widget.disable()

        if self._select_team is not None:
            if not self._team_is_selectable:
                self._select_team.disable()
        if self._select_workspace is not None:
            if not self._workspace_is_selectable:
                self._select_workspace.disable()

    def enable(self) -> None:
        """Enable the widget in the UI."""
        for widget in self._widgets:
            widget.enable()

        if self._select_team is not None:
            if not self._team_is_selectable:
                self._select_team.disable()
        if self._select_workspace is not None:
            if not self._workspace_is_selectable:
                self._select_workspace.disable()

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
        """Set the project ID to read datasets from.

        :param project_id: The ID of the project.
        :type project_id: int
        """
        if not self._compact:
            self._select_project.set_value(project_id)
        self._project_id = project_id
        self._select_dataset.set_items(self._read_datasets(project_id))

    def get_selected_project_id(self) -> Optional[int]:
        """Get the ID of the selected project.

        :return: The ID of the selected project.
        :rtype: Optional[int]
        """
        return self.project_id

    def set_dataset_id(self, dataset_id: int) -> None:
        """Set the ID of the dataset to be selected by default.

        :param id: The ID of the dataset.
        :type id: int
        """
        self._select_dataset.set_selected_by_id(dataset_id)

    def set_dataset_ids(self, dataset_ids: List[int]) -> None:
        """Set the IDs of the datasets to be selected by default.

        :raise ValueError: If multiselect is disabled.
        :param ids: The IDs of the datasets.
        :type ids: List[int]
        """
        if not self._multiselect:
            raise ValueError("This method can only be called when multiselect is enabled.")
        self._select_all_datasets_checkbox.uncheck()
        self._select_dataset.set_selected_by_id(dataset_ids)

    def team_changed(self, func: Callable) -> Callable:
        """Decorator to set the callback function for the team changed event."""
        if self._compact:
            raise ValueError("callback 'team_changed' is not available in compact mode.")
        self._team_changed_callbacks.append(func)
        return func

    def workspace_changed(self, func: Callable) -> Callable:
        """Decorator to set the callback function for the workspace changed event."""
        if self._compact:
            raise ValueError("callback 'workspace_changed' is not available in compact mode.")
        self._workspace_changed_callbacks.append(func)
        return func

    def project_changed(self, func: Callable) -> Callable:
        """Decorator to set the callback function for the project changed event."""
        if self._compact:
            raise ValueError("callback 'project_changed' is not available in compact mode.")
        self._project_changed_callbacks.append(func)
        return func

    def value_changed(self, func: Callable) -> Callable:
        """Decorator to set the callback function for the value changed event.

        :param func: The callback function.
        :type func: Callable
        :return: The callback function.
        :rtype: Callable
        """

        @self._select_dataset.value_changed
        def _click(items: Union[List[TreeSelect.Item], TreeSelect.Item]):
            if isinstance(items, list):
                res = [item.id for item in items]
            else:
                res = items.id

            func(res)

        return _click

    def _create_select_all_datasets_checkbox(self, select_all_datasets: bool) -> None:
        """Create the checkbox to select all datasets.

        :param select_all_datasets: Whether all datasets should be selected by default.
        :type select_all_datasets: bool
        """
        if not self._multiselect:
            # We'll only create the checkbox if multiselect is enabled.
            return
        select_all_datasets_checkbox = Checkbox("Select all datasets")

        @select_all_datasets_checkbox.value_changed
        def select_all_datasets_checkbox_handler(checked: bool) -> None:
            """Handler function for the event when the checkbox value changes.

            :param checked: The value of the checkbox.
            :type checked: bool
            """
            if self._project_id is None:
                return

            if checked:
                self._select_dataset.select_all()
                self._select_dataset_field.hide()
            else:
                self._select_dataset.clear_selected()
                self._select_dataset_field.show()

        if select_all_datasets:
            self._select_dataset_field.hide()
            select_all_datasets_checkbox.check()

        self._widgets.append(select_all_datasets_checkbox)
        self._select_all_datasets_checkbox = select_all_datasets_checkbox

    def _create_dataset_selector(
        self, flat: bool, always_open: bool, select_all_datasets: bool
    ) -> None:
        """Create the dataset selector.

        :param flat: Whether the dataset selector should be flat.
        :type flat: bool
        :param always_open: Whether the dataset selector should always be open.
        :type always_open: bool
        :param select_all_datasets: Whether all datasets should be selected by default.
        :type select_all_datasets: bool
        """
        self._select_dataset = TreeSelect(
            items=self._read_datasets(self._project_id),
            multiple_select=self._multiselect,
            flat=flat,
            always_open=always_open,
            append_to_body=self._append_to_body,
            width=self._width,
            placeholder="Select dataset",
        )
        if self._dataset_id is not None:
            self._select_dataset.set_selected_by_id(self._dataset_id)
        if select_all_datasets:
            self._select_dataset.select_all()
        self._select_dataset_field = Field(self._select_dataset, title="Dataset")

        # Adding the dataset selector to the list of widgets to be added to the container.
        self._widgets.append(self._select_dataset_field)

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

            for callback in self._team_changed_callbacks:
                callback(team_id)

        def workspace_selector_handler(workspace_id: int):
            """Handler function for the event when the workspace selector value changes.

            :param workspace_id: The ID of the selected workspace.
            :type workspace_id: int
            """
            self._select_project.set(items=self._get_select_items(workspace_id=workspace_id))
            self._workspace_id = workspace_id

            for callback in self._workspace_changed_callbacks:
                callback(workspace_id)

        def project_selector_handler(project_id: int):
            """Handler function for the event when the project selector value changes.

            :param project_id: The ID of the selected project.
            :type project_id: int
            """
            self._select_dataset.set_items(self._read_datasets(project_id))
            self._project_id = project_id

            if (
                self._select_all_datasets_checkbox is not None
                and self._select_all_datasets_checkbox.is_checked()
            ):
                self._select_dataset.select_all()
                self._select_dataset_field.hide()

            for callback in self._project_changed_callbacks:
                callback(project_id)

        self._select_team = Select(
            items=self._get_select_items(),
            placeholder="Select team",
            filterable=True,
            width_px=self._width,
        )
        self._select_team.set_value(self._team_id)
        if not team_is_selectable:
            self._select_team.disable()
        self._select_team_field = Field(self._select_team, title="Team")

        self._select_workspace = Select(
            items=self._get_select_items(team_id=self._team_id),
            placeholder="Select workspace",
            filterable=True,
            width_px=self._width,
        )
        self._select_workspace.set_value(self._workspace_id)
        if not workspace_is_selectable:
            self._select_workspace.disable()
        self._select_workspace_field = Field(self._select_workspace, title="Workspace")

        self._select_project = Select(
            items=self._get_select_items(workspace_id=self._workspace_id),
            placeholder="Select project",
            filterable=True,
            width_px=self._width,
        )
        self._select_project.set_value(self._project_id)
        self._select_project_field = Field(self._select_project, title="Project")

        # Register the event handlers._select_project
        self._select_team.value_changed(team_selector_handler)
        self._select_workspace.value_changed(workspace_selector_handler)
        self._select_project.value_changed(project_selector_handler)

        # Adding widgets to the list, so they can be added to the container.
        self._widgets.extend(
            [self._select_team_field, self._select_workspace_field, self._select_project_field]
        )

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

    def _read_datasets(self, project_id: Optional[int]) -> Optional[List[TreeSelect.Item]]:
        """Get the lisf of TreeSelect.Item objects representing the dataset hierarchy.

        :param project_id: The ID of the project.
        :type project_id: Optional[int]
        :return: The list of TreeSelect.Item objects.
        :rtype: Optional[List[TreeSelect.Item]]
        """
        if not project_id:
            return None
        dataset_tree = self._api.dataset.get_tree(project_id)

        def convert_tree_to_list(node, parent_id: Optional[int] = None):
            """
            Recursively converts a tree of DatasetInfo objects into a list of
                SelectDatasetTree.Item objects.

            :param node: The current node in the tree (a tuple of DatasetInfo and its children).
            :param parent_id: The ID of the parent dataset, if any.
            :return: A list of SelectDatasetTree.Item objects representing the tree.
            """
            result = []
            for dataset_info, children in node.items():
                item = TreeSelect.Item(
                    id=dataset_info.id,
                    label=dataset_info.name,
                    children=convert_tree_to_list(children, parent_id=dataset_info.id),
                )

                result.append(item)

            return result

        return convert_tree_to_list(dataset_tree)

    def set_project_id(self, project_id: Optional[int]) -> None:
        """Set the project ID to read datasets from.

        :param project_id: The ID of the project.
        :type project_id: int
        """
        self.project_id = project_id

    def _get_selected(self) -> Optional[Union[List[int], int]]:
        """Get the ID of the selected dataset(s).

        :return: The ID of the selected dataset(s).
        :rtype: Optional[Union[List[int], int]]
        """
        selected = self._select_dataset.get_selected()
        if not selected:
            return None

        if isinstance(selected, list):
            return [item.id for item in selected]
        else:
            return selected.id

    def get_selected_ids(self) -> Optional[List[int]]:
        """Get the IDs of the selected datasets.

        raise ValueError if multiselect is disabled.
        return: The IDs of the selected datasets.
        rtype: Optional[List[int]]
        """
        if not self._multiselect:
            raise ValueError("This method can only be called when multiselect is enabled.")
        return self._get_selected()

    def get_selected_id(self) -> Optional[int]:
        """Get the ID of the selected dataset.

        raise ValueError if multiselect is enabled.
        return: The ID of the selected dataset.
        rtype: Optional[int]
        """
        if self._multiselect:
            raise ValueError("This method can only be called when multiselect is disabled.")
        return self._get_selected()

    def is_all_selected(self) -> bool:
        """Check if all datasets are selected.

        return: True if all datasets are selected, False otherwise.
        rtype: bool
        """
        return self._select_dataset.is_all_selected()

    def select_all(self) -> None:
        """Select all datasets."""
        self._select_dataset.select_all()
