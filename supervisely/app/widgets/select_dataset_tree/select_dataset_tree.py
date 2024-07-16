from typing import Union

from supervisely.api.api import Api
from supervisely.app.widgets.tree_select.tree_select import TreeSelect


class SelectDatasetTree(TreeSelect):
    def __init__(
        self,
        default_id: Union[int, None] = None,
        project_id: Union[int, None] = None,
        multiselect: bool = False,
        compact: bool = False,
        flat: bool = False,
        always_open: bool = False,
        widget_id: Union[str, None] = None,
    ):
        self._api = Api()
        self._default_id = default_id
        self._project_id = project_id
        self._multiselect = multiselect
        self._compact = compact
        if not compact:
            raise NotImplementedError("Not compact mode is not supported yet")

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
            Recursively converts a tree of DatasetInfo objects into a list of dictionaries.

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
        self._project_id = project_id
        self._items = self.read_datasets(project_id)
