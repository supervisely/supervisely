from typing import Union

from supervisely.api.api import Api
from typing import List, Callable
from supervisely.app.widgets.tree_select.tree_select import TreeSelect


class SelectDatasetTree(TreeSelect):
    def __init__(
        self,
        default_id: Union[int, None] = None,
        project_id: Union[int, None] = None,
        multiselect: bool = False,
        flat: bool = False,
        always_open: bool = False,
        widget_id: Union[str, None] = None,
    ):
        self._api = Api()
        self._default_id = default_id
        self._project_id = project_id
        self._multiselect = multiselect

        if project_id is not None:
            items = self.read_datasets(project_id)
        else:
            items = []

        super().__init__(
            items=items,
            multiple_select=multiselect,
            flat=flat,
            always_open=always_open,
            widget_id=widget_id,
        )

    def read_datasets(self, project_id: int):
        dataset_tree = self._api.dataset.get_tree(project_id)

        def convert_tree_to_list(node, parent_id=None):
            """
            Recursively converts a tree of DatasetInfo objects into a list of
                SelectDatasetTree.Item objects.

            :param node: The current node in the tree (a tuple of DatasetInfo and its children).
            :param parent_id: The ID of the parent dataset, if any.
            :return: A list of dictionaries representing the dataset hierarchy.
            """
            result = []
            for dataset_info, children in node.items():
                item = SelectDatasetTree.Item(
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
        self._project_id = project_id
        self._items = self.read_datasets(project_id)

    def get_selected_ids(self) -> List[int]:
        """Get the IDs of the selected datasets. If multiselect is disabled, returns a list
            with a single ID.

        :return: The IDs of the selected datasets.
        :rtype: List[int]
        """
        res = self.get_selected()
        if not isinstance(res, list):
            res = [res]
        return [int(item.id) for item in res]

    def get_selected_id(self) -> Union[int, None]:
        """Get the ID of the selected dataset. Use only when multiselect is disabled.

        :raises ValueError: If multiple items are selected.
        :return: The ID of the selected dataset.
        :rtype: Union[int, None]
        """
        res = self.get_selected()
        if res is None:
            return None
        if isinstance(res, list):
            raise ValueError("Multiple items selected, while trying to get a single ID.")
        return int(res.id)

    def value_changed(self, func: Callable) -> Callable:
        """Decorator for the function to be called when the value is changed.
        Decorated function will receive a list of selected dataset IDs if multiselect is enabled,
        or a single dataset ID if multiselect is disabled.

        :param func: The function to be called when the value is changed.
        :type func: Callable
        :return: The decorated function.
        :rtype: Callable
        """
        route_path = self.get_route_path(TreeSelect.Routes.VALUE_CHANGED)
        server = self._sly_app.get_server()
        self._changes_handled = True

        @server.post(route_path)
        def _click():
            if self._multiselect:
                res = self.get_selected_ids()
            else:
                res = self.get_selected_id()

            func(res)

        return _click
