# coding: utf-8
"""create/download/update :class:`Dataset<supervisely.project.project.Dataset>`"""

# docs
from __future__ import annotations

import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from tqdm import tqdm

from supervisely._utils import (
    abs_url,
    compress_image_url,
    is_development,
    run_coroutine,
)
from supervisely.annotation.annotation import Annotation
from supervisely.api.module_api import (
    ApiField,
    ModuleApi,
    RemoveableModuleApi,
    UpdateableModule,
    _get_single_item,
)
from supervisely.io.json import load_json_file
from supervisely.project.project_type import ProjectType

if TYPE_CHECKING:
    from supervisely.project.project import ProjectMeta

from supervisely import logger


class DatasetInfo(NamedTuple):
    """Represent Supervisely Dataset information.

    :param id: Dataset ID.
    :type id: int
    :param name: Dataset Name.
    :type name: str
    :param description: Dataset description.
    :type description: str
    :param size: Dataset size.
    :type size: int
    :param project_id: Project ID in which the Dataset is located.
    :type project_id: int
    :param images_count: Number of images in the Dataset.
    :type images_count: int
    :param items_count: Number of items in the Dataset.
    :type items_count: int
    :param created_at: Date and time when the Dataset was created.
    :type created_at: str
    :param updated_at: Date and time when the Dataset was last updated.
    :type updated_at: str
    :param reference_image_url: URL of the reference image.
    :type reference_image_url: str
    :param team_id: Team ID in which the Dataset is located.
    :type team_id: int
    :param workspace_id: Workspace ID in which the Dataset is located.
    :type workspace_id: int
    :param parent_id: Parent Dataset ID. If the Dataset is not nested, then the value is None.
    :type parent_id: Union[int, None]
    """

    id: int
    name: str
    description: str
    size: int
    project_id: int
    images_count: int
    items_count: int
    created_at: str
    updated_at: str
    reference_image_url: str
    team_id: int
    workspace_id: int
    parent_id: Union[int, None]
    custom_data: dict
    created_by: int

    @property
    def image_preview_url(self):
        res = self.reference_image_url
        if is_development():
            res = abs_url(res)
        res = compress_image_url(url=res, height=200)
        return res


class DatasetApi(UpdateableModule, RemoveableModuleApi):
    """
    API for working with :class:`Dataset<supervisely.project.project.Dataset>`. :class:`DatasetApi<DatasetApi>` object is immutable.

    :param api: API connection to the server.
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Pass values into the API constructor (optional, not recommended)
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        project_id = 1951
        ds = api.dataset.get_list(project_id)
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple DatasetInfo information about Dataset.

        :Example:

         .. code-block:: python

            DatasetInfo(id=452984,
                        name='ds0',
                        description='',
                        size='3997776',
                        project_id=118909,
                        images_count=11,
                        items_count=11,
                        created_at='2021-03-03T15:54:08.802Z',
                        updated_at='2021-03-16T09:31:37.063Z',
                        reference_image_url='https://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/K/q/jf/...png'),
                        team_id=1,
                        workspace_id=2
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.SIZE,
            ApiField.PROJECT_ID,
            ApiField.IMAGES_COUNT,
            ApiField.ITEMS_COUNT,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.REFERENCE_IMAGE_URL,
            ApiField.TEAM_ID,
            ApiField.WORKSPACE_ID,
            ApiField.PARENT_ID,
            ApiField.CUSTOM_DATA,
            ApiField.CREATED_BY_ID[0][0],
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **DatasetInfo**.
        """
        return "DatasetInfo"

    def __init__(self, api):
        ModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(
        self,
        project_id: int,
        filters: Optional[List[Dict[str, str]]] = None,
        recursive: Optional[bool] = False,
        parent_id: Optional[int] = None,
        include_custom_data: Optional[bool] = False,
    ) -> List[DatasetInfo]:
        """
        Returns list of dataset in the given project, or list of nested datasets
        in the dataset with specified parent_id.
        To get list of all datasets including nested, recursive parameter should be set to True.
        Otherwise, the method will return only datasets in the top level.

        :param project_id: Project ID in which the Datasets are located.
        :type project_id: int
        :param filters: List of params to sort output Datasets.
        :type filters: List[dict], optional
        :param recursive: If True, returns all Datasets from the given Project including nested Datasets.
        :type recursive: bool, optional
        :param parent_id: Parent Dataset ID. If set to None, the search will be performed at the top level of the Project,
            otherwise the search will be performed in the specified Dataset.
        :type parent_id: Union[int, None], optional
        :param include_custom_data: If True, the response will include the `custom_data` field for each Dataset.
        :type include_custom_data: bool, optional
        :return: List of all Datasets with information for the given Project. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[DatasetInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()
            ds = api.dataset.get_list(project_id)

            print(ds)
            # Output: [
            #     DatasetInfo(id=2532,
            #                 name="lemons",
            #                 description="",
            #                 size="861069",
            #                 project_id=1951,
            #                 images_count=6,
            #                 items_count=6,
            #                 created_at="2021-03-02T10:04:33.973Z",
            #                 updated_at="2021-03-10T09:31:50.341Z",
            #                 reference_image_url="http://app.supervisely.com/z6ut6j8bnaz1vj8aebbgs4-public/images/original/...jpg"),
            #                 DatasetInfo(id=2557,
            #                 name="kiwi",
            #                 description="",
            #                 size="861069",
            #                 project_id=1951,
            #                 images_count=6,
            #                 items_count=6,
            #                 created_at="2021-03-10T09:31:33.701Z",
            #                 updated_at="2021-03-10T09:31:44.196Z",
            #                 reference_image_url="http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg")
            # ]
        """
        if filters is None:
            filters = []

        if parent_id is not None:
            filters.append({"field": ApiField.PARENT_ID, "operator": "=", "value": parent_id})
            recursive = True

        method = "datasets.list"
        data = {
            ApiField.PROJECT_ID: project_id,
            ApiField.FILTER: filters,
            ApiField.RECURSIVE: recursive,
        }
        if include_custom_data:
            data[ApiField.EXTRA_FIELDS] = [ApiField.CUSTOM_DATA]

        return self.get_list_all_pages(method, data)

    def get_info_by_id(self, id: int, raise_error: Optional[bool] = False) -> DatasetInfo:
        """
        Get Datasets information by ID.

        :param id: Dataset ID in Supervisely.
        :type id: int
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            dataset_id = 384126

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_info = api.dataset.get_info_by_id(dataset_id)
        """
        info = self._get_info_by_id(id, "datasets.info")
        if info is None and raise_error is True:
            raise KeyError(f"Dataset with id={id} not found in your account")
        return info

    def _get_effective_new_name(
        self, project_id: int, name: str, change_name_if_conflict: bool, parent_id: int = None
    ):
        return (
            self._get_free_name(
                exist_check_fn=lambda name: self.get_info_by_name(
                    project_id, name, parent_id=parent_id
                )
                is not None,
                name=name,
            )
            if change_name_if_conflict
            else name
        )

    def create(
        self,
        project_id: int,
        name: str,
        description: Optional[str] = "",
        change_name_if_conflict: Optional[bool] = False,
        parent_id: Optional[int] = None,
        custom_data: Optional[Dict[Any, Any]] = None,
    ) -> DatasetInfo:
        """
        Create Dataset with given name in the given Project.

        :param project_id: Project ID in Supervisely where Dataset will be created.
        :type project_id: int
        :param name: Dataset Name.
        :type name: str
        :param description: Dataset description.
        :type description: str, optional
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param parent_id: Parent Dataset ID. If set to None, then the Dataset will be created at
            the top level of the Project, otherwise the Dataset will be created in a specified Dataset.
        :type parent_id: Union[int, None]
        :param custom_data: Custom data to store in the Dataset.
        :type custom_data: Dict[Any, Any], optional
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 116482

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_info = api.dataset.get_list(project_id)
            print(len(ds_info)) # 1

            new_ds = api.dataset.create(project_id, 'new_ds')
            new_ds_info = api.dataset.get_list(project_id)
            print(len(new_ds_info)) # 2
        """
        effective_name = self._get_effective_new_name(
            project_id=project_id,
            name=name,
            change_name_if_conflict=change_name_if_conflict,
            parent_id=parent_id,
        )
        method = "datasets.add"
        payload = {
            ApiField.PROJECT_ID: project_id,
            ApiField.NAME: effective_name,
            ApiField.DESCRIPTION: description,
            ApiField.PARENT_ID: parent_id,
        }
        if custom_data is not None:
            payload[ApiField.CUSTOM_DATA] = custom_data
        response = self._api.post(method, payload)
        return self._convert_json_info(response.json())

    def get_or_create(
        self,
        project_id: int,
        name: str,
        description: Optional[str] = "",
        parent_id: Optional[int] = None,
    ) -> DatasetInfo:
        """
        Checks if Dataset with given name already exists in the Project, if not creates Dataset with the given name.
        If parent id is specified then the search will be performed in the specified Dataset, otherwise
        the search will be performed at the top level of the Project.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param name: Dataset name.
        :type name: str
        :param description: Dataset description.
        :type description: str, optional
        :param parent_id: Parent Dataset ID. If set to None, then the Dataset will be created at
            the top level of the Project, otherwise the Dataset will be created in a specified Dataset.
        :type parent_id: Union[int, None]
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 116482

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_info = api.dataset.get_list(project_id)
            print(len(ds_info)) # 1

            api.dataset.get_or_create(project_id, 'ds1')
            ds_info = api.dataset.get_list(project_id)
            print(len(ds_info)) # 1

            api.dataset.get_or_create(project_id, 'new_ds')
            ds_info = api.dataset.get_list(project_id)
            print(len(ds_info)) # 2
        """
        dataset_info = self.get_info_by_name(project_id, name, parent_id=parent_id)
        if dataset_info is None:
            dataset_info = self.create(
                project_id, name, description=description, parent_id=parent_id
            )
        return dataset_info

    def update(
        self,
        id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        custom_data: Optional[Dict[Any, Any]] = None,
    ) -> DatasetInfo:
        """Update Dataset information by given ID.

        :param id: Dataset ID in Supervisely.
        :type id: int
        :param name: New Dataset name.
        :type name: str, optional
        :param description: New Dataset description.
        :type description: str, optional
        :param custom_data: New custom data.
        :type custom_data: Dict[Any, Any], optional
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`

        :Usage example:

             .. code-block:: python

                import supervisely as sly

                dataset_id = 384126

                os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
                os.environ['API_TOKEN'] = 'Your Supervisely API Token'
                api = sly.Api.from_env()

                new_ds = api.dataset.update(dataset_id, name='new_ds', description='new description')
        """
        fields = [name, description, custom_data]  # Extend later if needed.
        if all(f is None for f in fields):
            raise ValueError(f"At least one of the fields must be specified: {fields}")

        payload = {
            ApiField.ID: id,
            ApiField.NAME: name,
            ApiField.DESCRIPTION: description,
            ApiField.CUSTOM_DATA: custom_data,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        response = self._api.post(self._get_update_method(), payload)
        return self._convert_json_info(response.json())

    def update_custom_data(self, id: int, custom_data: Dict[Any, Any]) -> DatasetInfo:
        """Update custom data for Dataset by given ID.
        Custom data is a dictionary that can store any additional information about the Dataset.

        :param id: Dataset ID in Supervisely.
        :type id: int
        :param custom_data: New custom data.
        :type custom_data: Dict[Any, Any]
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            dataset_id = 384126

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_ds = api.dataset.update_custom_data(dataset_id, custom_data={'key': 'value'})
        """
        return self.update(id, custom_data=custom_data)

    def _get_update_method(self):
        """ """
        return "datasets.editInfo"

    def _remove_api_method_name(self):
        """ """
        return "datasets.remove"

    def copy_batch(
        self,
        dst_project_id: int,
        ids: List[int],
        new_names: Optional[List[str]] = None,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> List[DatasetInfo]:
        """
        Copy given Datasets to the destination Project by IDs.

        :param dst_project_id: Destination Project ID in Supervisely.
        :type dst_project_id: int
        :param ids: IDs of copied Datasets.
        :type ids: List[int]
        :param new_names: New Datasets names.
        :type new_names: List[str], optional
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True copies Datasets with annotations, otherwise copies just items from Datasets without annotations.
        :type with_annotations: bool, optional
        :raises: :class:`RuntimeError` if can not match "ids" and "new_names" lists, len(ids) != len(new_names)
        :return: Information about Datasets. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[DatasetInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_proj_id = 1980
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 0

            ds_ids = [2532, 2557]
            ds_names = ["lemon_test", "kiwi_test"]

            copied_datasets = api.dataset.copy_batch(dst_proj_id, ids=ds_ids, new_names=ds_names, with_annotations=True)
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 2
        """
        if new_names is not None and len(ids) != len(new_names):
            raise RuntimeError(
                'Can not match "ids" and "new_names" lists, len(ids) != len(new_names)'
            )
        project_info = self._api.project.get_info_by_id(dst_project_id)
        if project_info.type == str(ProjectType.IMAGES):
            items_api = self._api.image
        elif project_info.type == str(ProjectType.VIDEOS):
            items_api = self._api.video
        else:
            raise RuntimeError(f"Unsupported project type: {project_info.type}")

        new_datasets = []
        for idx, dataset_id in enumerate(ids):
            dataset = self.get_info_by_id(dataset_id)
            new_dataset_name = dataset.name
            if new_names is not None:
                new_dataset_name = new_names[idx]
            src_items = items_api.get_list(dataset.id)
            src_item_ids = [item.id for item in src_items]
            new_dataset = self._api.dataset.create(
                dst_project_id,
                new_dataset_name,
                dataset.description,
                change_name_if_conflict=change_name_if_conflict,
                custom_data=dataset.custom_data,
            )
            items_api.copy_batch(
                new_dataset.id, src_item_ids, change_name_if_conflict, with_annotations
            )
            new_datasets.append(new_dataset)
        return new_datasets

    def copy(
        self,
        dst_project_id: int,
        id: int,
        new_name: Optional[str] = None,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> DatasetInfo:
        """
        Copies given Dataset in destination Project by ID.

        :param dst_project_id: Destination Project ID in Supervisely.
        :type dst_project_id: int
        :param id: ID of copied Dataset.
        :type id: int
        :param new_name: New Dataset name.
        :type new_name: str, optional
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True copies Dataset with annotations, otherwise copies just items from Dataset without annotation.
        :type with_annotations: bool, optional
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_proj_id = 1982
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 0

            new_ds = api.dataset.copy(dst_proj_id, id=2540, new_name="banana", with_annotations=True)
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 1
        """
        new_datasets = self.copy_batch(
            dst_project_id, [id], [new_name], change_name_if_conflict, with_annotations
        )
        if len(new_datasets) == 0:
            return None
        return new_datasets[0]

    def move_batch(
        self,
        dst_project_id: int,
        ids: List[int],
        new_names: Optional[List[str]] = None,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> List[DatasetInfo]:
        """
        Moves given Datasets to the destination Project by IDs.

        :param dst_project_id: Destination Project ID in Supervisely.
        :type dst_project_id: int
        :param ids: IDs of moved Datasets.
        :type ids: List[int]
        :param new_names: New Datasets names.
        :type new_names: List[str], optional
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True moves Datasets with annotations, otherwise moves just items from Datasets without annotations.
        :type with_annotations: bool, optional
        :raises: :class:`RuntimeError` if can not match "ids" and "new_names" lists, len(ids) != len(new_names)
        :return: Information about Datasets. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[DatasetInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_proj_id = 1978
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 0

            ds_ids = [2545, 2560]
            ds_names = ["banana_test", "mango_test"]

            movied_datasets = api.dataset.move_batch(dst_proj_id, ids=ds_ids, new_names=ds_names, with_annotations=True)
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 2
        """
        new_datasets = self.copy_batch(
            dst_project_id, ids, new_names, change_name_if_conflict, with_annotations
        )
        self.remove_batch(ids)
        return new_datasets

    def move(
        self,
        dst_project_id: int,
        id: int,
        new_name: Optional[str] = None,
        change_name_if_conflict: Optional[bool] = False,
        with_annotations: Optional[bool] = False,
    ) -> DatasetInfo:
        """
        Moves given Dataset in destination Project by ID.

        :param dst_project_id: Destination Project ID in Supervisely.
        :type dst_project_id: int
        :param id: ID of moved Dataset.
        :type id: int
        :param new_name: New Dataset name.
        :type new_name: str, optional
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :param with_annotations: If True moves Dataset with annotations, otherwise moves just items from Dataset without annotation.
        :type with_annotations: bool, optional
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            dst_proj_id = 1985
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 0

            new_ds = api.dataset.move(dst_proj_id, id=2550, new_name="cucumber", with_annotations=True)
            ds = api.dataset.get_list(dst_proj_id)
            print(len(ds)) # 1
        """
        new_dataset = self.copy(
            dst_project_id, id, new_name, change_name_if_conflict, with_annotations
        )
        self.remove(id)
        return new_dataset

    def move_to_dataset(self, dataset_id: int, destination_dataset_id: int) -> None:
        """Moves dataset with specified ID to the dataset with specified destination ID.

        :param dataset_id: ID of the dataset to be moved.
        :type dataset_id: int
        :param destination_dataset_id: ID of the destination dataset.
        :type destination_dataset_id: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            dataset_id = 123
            destination_dataset_id = 456

            api.dataset.move_to_dataset(dataset_id, destination_dataset_id)
        """
        self._api.post(
            "datasets.move", {ApiField.SRC_ID: dataset_id, ApiField.DEST_ID: destination_dataset_id}
        )

    def _convert_json_info(self, info: dict, skip_missing=True):
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.reference_image_url is not None:
            res = res._replace(reference_image_url=res.reference_image_url)
        if res.items_count is None:
            res = res._replace(items_count=res.images_count)
        return DatasetInfo(**res._asdict())

    def remove_permanently(
        self, ids: Union[int, List], batch_size: int = 50, progress_cb=None
    ) -> List[dict]:
        """
        !!! WARNING !!!
        Be careful, this method deletes data from the database, recovery is not possible.

        Delete permanently datasets with given IDs from the Supervisely server.
        All dataset IDs must belong to the same team.
        Therefore, it is necessary to sort IDs before calling this method.


        :param ids: IDs of datasets in Supervisely.
        :type ids: Union[int, List]
        :param batch_size: The number of entities that will be deleted by a single API call. This value must be in the range 1-50 inclusive, if you set a value out of range it will automatically adjust to the boundary values.
        :type batch_size: int, optional
        :param progress_cb: Function for control delete progress.
        :type progress_cb: Callable, optional
        :return: A list of response content in JSON format for each API call.
        :rtype: List[dict]
        """
        if batch_size > 50:
            batch_size = 50
        elif batch_size < 1:
            batch_size = 1

        if isinstance(ids, int):
            datasets = [{ApiField.ID: ids}]
        else:
            datasets = [{ApiField.ID: id} for id in ids]

        batches = [datasets[i : i + batch_size] for i in range(0, len(datasets), batch_size)]
        responses = []
        for batch in batches:
            response = self._api.post("datasets.remove.permanently", {ApiField.DATASETS: batch})
            if progress_cb is not None:
                progress_cb(len(batch))
            responses.append(response.json())
        return responses

    def get_list_all(
        self,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = None,
        sort_order: Optional[str] = None,
        per_page: Optional[int] = None,
        page: Union[int, Literal["all"]] = "all",
        include_custom_data: Optional[bool] = False,
    ) -> dict:
        """
        List all available datasets from all available teams for the user that match the specified filtering criteria.

        :param filters: List of parameters for filtering the available Datasets.
                        Every Dict must consist of keys:
                        - 'field': Takes values 'id', 'projectId', 'workspaceId', 'groupId', 'createdAt', 'updatedAt'
                        - 'operator': Takes values '=', 'eq', '!=', 'not', 'in', '!in', '>', 'gt', '>=', 'gte', '<', 'lt', '<=', 'lte'
                        - 'value': Takes on values according to the meaning of 'field' or null
        :type filters: List[Dict[str, str]], optional
        :param sort: Specifies by which parameter to sort the project list.
                        Takes values 'id', 'name', 'size', 'createdAt', 'updatedAt'
        :type sort: str, optional
        :param sort_order: Determines which value to list from.
        :type sort_order: str, optional
        :param per_page: Number of first items found to be returned.
                        'None' will return the first page with a default size of 20000 datasets.
        :type per_page: int, optional
        :param page: Page number, used to retrieve the following items if the number of them found is more than per_page.
                     The default value is 'all', which retrieves all available datasets.
                     'None' will return the first page with datasets, the amount of which is set in param 'per_page'.
        :type page: Union[int, Literal["all"]], optional
        :param include_custom_data: If True, the response will include the `custom_data` field for each Dataset.
        :type include_custom_data: bool, optional

        :return: Search response information and 'DatasetInfo' of all datasets that are searched by a given criterion.
        :rtype: dict

        :Usage example:

        .. code-block:: python

            import supervisely as sly
            import os

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            filter_1 = {
                "field": "updatedAt",
                "operator": "<",
                "value": "2023-12-03T14:53:00.952Z"
            }
            filter_2 = {
                "field": "updatedAt",
                "operator": ">",
                "value": "2023-04-03T14:53:00.952Z"
            }
            filters = [filter_1, filter_2]
            datasets = api.dataset.get_list_all(filters)
            print(datasets)
            # Output:
            # {
            #     "total": 2,
            #     "perPage": 20000,
            #     "pagesCount": 1,
            #     "entities": [ DatasetInfo(id = 16,
            #                       name = 'ds1',
            #                       description = None,
            #                       size = '861069',
            #                       project_id = 22,
            #                       images_count = None,
            #                       items_count = None,
            #                       created_at = '2020-04-03T13:43:24.000Z',
            #                       updated_at = '2020-04-03T14:53:00.952Z',
            #                       reference_image_url = None,
            #                       team_id = 2,
            #                       workspace_id = 2),
            #                   DatasetInfo(id = 17,
            #                       name = 'ds1',
            #                       description = None,
            #                       size = '1177212',
            #                       project_id = 23,
            #                       images_count = None,
            #                       items_count = None,
            #                       created_at = '2020-04-03T13:43:24.000Z',
            #                       updated_at = '2020-04-03T14:53:00.952Z',
            #                       reference_image_url = None,
            #                       team_id = 2,
            #                       workspace_id = 2
            #                       )
            #                 ]
            # }

        """

        method = "datasets.list.all"

        request_body = {}
        if filters is not None:
            request_body[ApiField.FILTER] = filters
        if sort is not None:
            request_body[ApiField.SORT] = sort
        if sort_order is not None:
            request_body[ApiField.SORT_ORDER] = sort_order
        if per_page is not None:
            request_body[ApiField.PER_PAGE] = per_page
        if page is not None and page != "all":
            request_body[ApiField.PAGE] = page
        if include_custom_data:
            request_body[ApiField.EXTRA_FIELDS] = [ApiField.CUSTOM_DATA]

        first_response = self._api.post(method, request_body).json()

        total = first_response.get(ApiField.TOTAL)
        per_page = first_response.get("perPage")
        pages_count = first_response.get("pagesCount")

        def _convert_entities(response_dict: dict):
            """
            Convert entities dict to DatasetInfo
            """
            response_dict[ApiField.ENTITIES] = [
                self._convert_json_info(item) for item in response_dict[ApiField.ENTITIES]
            ]

        if page == "all":
            results = first_response["entities"]
            if pages_count == 1 and len(results) == total:
                pass
            else:
                for page_idx in range(2, pages_count + 1):
                    temp_resp = self._api.post(
                        method, {**request_body, "page": page_idx, "per_page": per_page}
                    )
                    temp_items = temp_resp.json()["entities"]
                    results.extend(temp_items)

                if len(results) != total:
                    raise RuntimeError(
                        "Method {!r}: error during pagination, some items are missed".format(method)
                    )
            pass
        elif page is None or page <= pages_count:
            pass
        else:
            raise RuntimeError(
                f"Method {method}: error during pagination, some items are missed. Number of total pages [{pages_count}] is less than the page number requested [{page}]."
            )
        _convert_entities(first_response)
        return first_response

    def get_info_by_name(
        self, project_id: int, name: str, fields: List[str] = None, parent_id: Optional[int] = None
    ) -> Union[DatasetInfo, None]:
        """Return Dataset information by name or None if Dataset does not exist.
        If parent_id is not None, the search will be performed in the specified Dataset.
        Otherwise the search will be performed at the top level of the Project.

        :param project_id: Project ID in which the Dataset is located.
        :type project_id: int
        :param name: Dataset name.
        :type name: str
        :param fields: List of fields to return. If None, then all fields are returned.
        :type fields: List[str], optional
        :param parent_id: Parent Dataset ID. If the Dataset is not nested, then the value is None.
        :type parent_id: Union[int, None]
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: Union[DatasetInfo, None]
        """
        filters = [{"field": ApiField.NAME, "operator": "=", "value": name}]
        items = self.get_list(project_id, filters, parent_id=parent_id)
        return _get_single_item(items)

    def get_tree(self, project_id: int) -> Dict[DatasetInfo, Dict]:
        """Returns a tree of all datasets in the project as a dictionary,
        where the keys are the DatasetInfo objects and the values are dictionaries
        containing the children of the dataset.
        Recommended to use with the dataset_tree method to iterate over the tree.

        :param project_id: Project ID for which the tree is built.
        :type project_id: int
        :return: Dictionary of datasets and their children.
        :rtype: Dict[DatasetInfo, Dict]
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123

            dataset_tree = api.dataset.get_tree(project_id)
            print(dataset_tree)
            # Output:
            # {
            #     DatasetInfo(id=2532, name="lemons", description="", ...: {
            #         DatasetInfo(id=2557, name="kiwi", description="", ...: {}
            #     }
            # }
        """

        datasets = self.get_list(project_id, recursive=True)
        dataset_tree = {}

        def build_tree(parent_id):
            children = {}
            for dataset in datasets:
                if dataset.parent_id == parent_id:
                    children[dataset] = build_tree(dataset.id)
            return children

        for dataset in datasets:
            if dataset.parent_id is None:
                dataset_tree[dataset] = build_tree(dataset.id)

        return dataset_tree

    def _yield_tree(
        self, tree: Dict[DatasetInfo, Dict], path: List[str]
    ) -> Generator[Tuple[List[str], DatasetInfo], None, None]:
        """
        Helper method for recursive tree traversal.
        Yields tuples of (path, dataset) for all datasets in the tree. For each node (dataset) at the current level,
        yields its (path, dataset) before recursively traversing and yielding from its children.
        
        :param tree: Tree structure to yield from.
        :type tree: Dict[DatasetInfo, Dict]
        :param path: Current path (used for recursion).
        :type path: List[str]
        :return: Generator of tuples of (path, dataset).
        :rtype: Generator[Tuple[List[str], DatasetInfo], None, None]
        """
        for dataset, children in tree.items():
            yield path, dataset
            new_path = path + [dataset.name]
            if children:
                yield from self._yield_tree(children, new_path)

    def _find_dataset_in_tree(
        self, tree: Dict[DatasetInfo, Dict], target_id: int, path: List[str] = None
    ) -> Tuple[Optional[DatasetInfo], Optional[Dict], List[str]]:
        """Find a specific dataset in the tree and return its subtree and path.
        
        :param tree: Tree structure to search in.
        :type tree: Dict[DatasetInfo, Dict]
        :param target_id: ID of the dataset to find.
        :type target_id: int
        :param path: Current path (used for recursion).
        :type path: List[str], optional
        :return: Tuple of (found_dataset, its_subtree, path_to_dataset).
        :rtype: Tuple[Optional[DatasetInfo], Optional[Dict], List[str]]
        """
        if path is None:
            path = []
            
        for dataset, children in tree.items():
            if dataset.id == target_id:
                return dataset, children, path
            # Search in children
            if children:
                found_dataset, found_children, found_path = self._find_dataset_in_tree(
                    children, target_id, path + [dataset.name]
                )
                if found_dataset is not None:
                    return found_dataset, found_children, found_path
        return None, None, []

    def tree(self, project_id: int, dataset_id: Optional[int] = None) -> Generator[Tuple[List[str], DatasetInfo], None, None]:
        """Yields tuples of (path, dataset) for all datasets in the project.
        Path of the dataset is a list of parents, e.g. ["ds1", "ds2", "ds3"].
        For root datasets, the path is an empty list.

        :param project_id: Project ID in which the Dataset is located.
        :type project_id: int
        :param dataset_id: Optional Dataset ID to start the tree from. If provided, only yields
            the subtree starting from this dataset (including the dataset itself and all its children).
        :type dataset_id: Optional[int]
        :return: Generator of tuples of (path, dataset).
        :rtype: Generator[Tuple[List[str], DatasetInfo], None, None]
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123

            # Get all datasets in the project
            for parents, dataset in api.dataset.tree(project_id):
                parents: List[str]
                dataset: sly.DatasetInfo
                print(parents, dataset.name)

            # Get only a specific branch starting from dataset_id = 456
            for parents, dataset in api.dataset.tree(project_id, dataset_id=456):
                parents: List[str]
                dataset: sly.DatasetInfo
                print(parents, dataset.name)

            # Output:
            # [] ds1
            # ["ds1"] ds2
            # ["ds1", "ds2"] ds3
        """

        full_tree = self.get_tree(project_id)
        
        if dataset_id is None:
            # Return the full tree
            yield from self._yield_tree(full_tree, [])
        else:
            # Find the specific dataset and return only its subtree
            target_dataset, subtree, dataset_path = self._find_dataset_in_tree(full_tree, dataset_id)
            if target_dataset is not None:
                # Yield the target dataset first, then its children
                yield dataset_path, target_dataset
                if subtree:
                    new_path = dataset_path + [target_dataset.name]
                    yield from self._yield_tree(subtree, new_path)

    def get_nested(self, project_id: int, dataset_id: int) -> List[DatasetInfo]:
        """Returns a list of all nested datasets in the specified dataset.

        :param project_id: Project ID in which the Dataset is located.
        :type project_id: int
        :param dataset_id: Dataset ID for which the nested datasets are returned.
        :type dataset_id: int

        :return: List of nested datasets.
        :rtype: List[DatasetInfo]

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123
            dataset_id = 456

            datasets = api.dataset.get_nested(project_id, dataset_id)
            for dataset in datasets:
                print(dataset.name, dataset.id) # Output: ds1 123

        """
        tree = self.get_tree(project_id)

        nested = []

        def recurse(tree: Dict[DatasetInfo, Dict], needed_dataset: bool = False):
            for dataset_info, children in tree.items():
                if needed_dataset:
                    nested.append(dataset_info)

                recurse(children, needed_dataset or dataset_info.id == dataset_id)

        recurse(tree)
        return nested

    def exists(self, project_id: int, name: str, parent_id: int = None) -> bool:
        """
        Checks if the dataset with the given name exists in the project.
        If parent_id is not None, the search will be performed in the specified Dataset.

        :param project_id: Project ID in which the Dataset is located.
        :type project_id: int
        :param name: Dataset name.
        :type name: str
        :param parent_id: Parent Dataset ID. If the Dataset is not nested, then the value is None.
        :type parent_id: Union[int, None]
        :return: True if the dataset exists, False otherwise.
        :rtype: bool
        """
        return self.get_info_by_name(project_id, name, parent_id=parent_id) is not None

    def quick_import(
        self,
        dataset: Union[int, DatasetInfo],
        blob_path: str,
        offsets_path: str,
        anns: List[str],
        project_meta: Optional[ProjectMeta] = None,
        project_type: Optional[ProjectType] = None,
        log_progress: bool = True,
    ):
        """
        Quick import of images and annotations to the dataset.
        Used only for extended Supervisely format with blobs.
        Project will be automatically marked as blob project.

        IMPORTANT: Number of annotations must be equal to the number of images in offset file.
                   Image names in the offset file and annotation files must match.

        :param dataset: Dataset ID or DatasetInfo object.
        :type dataset: Union[int, DatasetInfo]
        :param blob_path: Local path to the blob file.
        :type blob_path: str
        :param offsets_path: Local path to the offsets file.
        :type offsets_path: str
        :param anns: List of annotation paths.
        :type anns: List[str]
        :param project_meta: ProjectMeta object.
        :type project_meta: Optional[ProjectMeta], optional
        :param project_type: Project type.
        :type project_type: Optional[ProjectType], optional
        :param log_progress: If True, show progress bar.
        :type log_progress: bool, optional


        :Usage example:

        .. code-block:: python

            import supervisely as sly
            from supervisely.project.project_meta import ProjectMeta
            from supervisely.project.project_type import ProjectType

            api = sly.Api.from_env()

            dataset_id = 123
            workspace_id = 456
            blob_path = "/path/to/blob"
            offsets_path = "/path/to/offsets"
            project_meta_path = "/path/to/project_meta.json"
            anns = ["/path/to/ann1.json", "/path/to/ann2.json", ...]

            # Create a new project, dataset and update its meta
            project = api.project.create(
                workspace_id,
                "Quick Import",
                type=sly.ProjectType.IMAGES,
                change_name_if_conflict=True,
            )
            dataset = api.dataset.create(project.id, "ds1")
            project_meta_json = sly.json.load_json_file(project_meta_path)
            meta = api.project.update_meta(project.id, meta=project_meta_json)

            dataset_info = api.dataset.quick_import(
                dataset=dataset.id,
                blob_path=blob_path,
                offsets_path=offsets_path,
                anns=anns,
                project_meta=ProjectMeta(),
                project_type=ProjectType.IMAGES,
                log_progress=True
            )

        """
        from supervisely.api.api import Api, ApiContext
        from supervisely.api.image_api import _BLOB_TAG_NAME
        from supervisely.project.project import TF_BLOB_DIR, ProjectMeta

        def _ann_objects_generator(ann_paths, project_meta):
            for ann in ann_paths:
                ann_json = load_json_file(ann)
                yield Annotation.from_json(ann_json, project_meta)

        self._api: Api

        if isinstance(dataset, int):
            dataset = self.get_info_by_id(dataset)

        project_info = self._api.project.get_info_by_id(dataset.project_id)

        if project_meta is None:
            meta_dict = self._api.project.get_meta(dataset.project_id)
            project_meta = ProjectMeta.from_json(meta_dict)

        if project_type is None:
            project_type = project_info.type

        if project_type != ProjectType.IMAGES:
            raise NotImplementedError(
                f"Quick import is not implemented for project type {project_type}"
            )

        # Set optimization context
        with ApiContext(
            api=self._api,
            project_id=dataset.project_id,
            dataset_id=dataset.id,
            project_meta=project_meta,
        ):
            dst_blob_path = os.path.join(f"/{TF_BLOB_DIR}", os.path.basename(blob_path))
            dst_offset_path = os.path.join(f"/{TF_BLOB_DIR}", os.path.basename(offsets_path))
            if log_progress:
                sizeb = os.path.getsize(blob_path) + os.path.getsize(offsets_path)
                b_progress_cb = tqdm(
                    total=sizeb,
                    unit="B",
                    unit_scale=True,
                    desc=f"Uploading blob to file storage",
                )
            else:
                b_progress_cb = None

            self._api.file.upload_bulk_fast(
                team_id=project_info.team_id,
                src_paths=[blob_path, offsets_path],
                dst_paths=[dst_blob_path, dst_offset_path],
                progress_cb=b_progress_cb.update,
            )

            blob_file_id = self._api.file.get_info_by_path(project_info.team_id, dst_blob_path).id

            if log_progress:
                of_progress_cb = tqdm(desc=f"Uploading images by offsets", total=len(anns)).update
            else:
                of_progress_cb = None

            image_info_generator = self._api.image.upload_by_offsets_generator(
                dataset=dataset,
                team_file_id=blob_file_id,
                progress_cb=of_progress_cb,
            )

            ann_map = {Path(ann).stem: ann for ann in anns}

            for image_info_batch in image_info_generator:
                img_ids = [img_info.id for img_info in image_info_batch]
                img_names = [img_info.name for img_info in image_info_batch]
                img_anns = [ann_map[img_name] for img_name in img_names]
                ann_objects = _ann_objects_generator(img_anns, project_meta)
                coroutine = self._api.annotation.upload_anns_async(
                    image_ids=img_ids, anns=ann_objects, log_progress=log_progress
                )
                run_coroutine(coroutine)
        try:
            custom_data = self._api.project.get_custom_data(dataset.project_id)
            custom_data[_BLOB_TAG_NAME] = True
            self._api.project.update_custom_data(dataset.project_id, custom_data)
        except:
            logger.warning("Failed to set blob tag for project")
