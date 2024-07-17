from typing import Callable, Dict, List, Optional, Union

import supervisely.io.env as env
from supervisely.api.api import Api
from supervisely.app.widgets import Widget
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.select.select import Select
from supervisely.app.widgets.tree_select.tree_select import TreeSelect
from supervisely.project.project_type import ProjectType


class SelectDatasetTree(Widget):

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
        team_is_selectable: bool = False,
        workspace_is_selectable: bool = False,
        widget_id: Union[str, None] = None,
    ):
        self._api = Api()

        if default_id is not None and project_id is None:
            raise ValueError("Project ID must be provided when default dataset ID is set.")

        self._team_id = env.team_id()
        self._workspace_id = env.workspace_id()
        self._project_id = project_id
        self._dataset_id = default_id

        self._multiselect = multiselect
        self._compact = compact

        # Extract values from Enum to match the .type property of the ProjectInfo object.
        self._project_types = (
            [project_type.value for project_type in allowed_project_types]
            if allowed_project_types is not None
            else None
        )

        # Widget components.
        self._select_team = None
        self._select_workspace = None
        self._select_project = None
        self._select_dataset = None

        # List of widgets will be used to create a Container.
        self._widgets = []

        if not compact:
            # If the widget is not compact, create team, workspace, and project selectors.
            self._create_selectors(team_is_selectable, workspace_is_selectable)

        # Create the dataset selector.
        self._create_dataset_selector(flat, always_open, select_all_datasets)

        # Group the selectors and the dataset selector into a container.
        self._content = Container(self._widgets)
        super().__init__(widget_id=widget_id, file_path=__file__)

    @property
    def team_id(self) -> int:
        """The ID of the team selected in the widget.

        :return: The ID of the team.
        :rtype: int
        """
        return self._team_id

    @property
    def workspace_id(self) -> int:
        """The ID of the workspace selected in the widget.

        :return: The ID of the workspace.
        :rtype: int
        """
        return self._workspace_id

    @property
    def project_id(self) -> Optional[int]:
        """The ID of the project selected in the widget.

        :return: The ID of the project.
        :rtype: Optional[int]
        """
        return self._project_id

    @project_id.setter
    def project_id(self, project_id: int) -> None:
        """Set the project ID to read datasets from.

        :param project_id: The ID of the project.
        :type project_id: int
        """
        if not self._compact:
            self._select_project.set_value(project_id)
        self._project_id = project_id
        self._select_dataset.set_items(self._read_datasets(project_id))

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
            width=193,
        )
        if self._dataset_id is not None:
            self._select_dataset.set_selected_by_id(self._dataset_id)
        if select_all_datasets:
            self._select_dataset.select_all()

        # Adding the dataset selector to the list of widgets to be added to the container.
        self._widgets.append(self._select_dataset)

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
            self._select_dataset.set_items(self._read_datasets(project_id))
            self._project_id = project_id

        self._select_team = Select(
            items=self._get_select_items(),
        )
        self._select_team.set_value(self._team_id)
        if not team_is_selectable:
            self._select_team.disable()

        self._select_workspace = Select(
            items=self._get_select_items(team_id=self._team_id),
            filterable=True,
        )
        self._select_workspace.set_value(self._workspace_id)
        if not workspace_is_selectable:
            self._select_workspace.disable()

        self._select_project = Select(
            items=self._get_select_items(workspace_id=self._workspace_id),
            filterable=True,
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
            :return: A list of dictionaries representing the dataset hierarchy.
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

    def set_project_id(self, project_id: int) -> None:
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
