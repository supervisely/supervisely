# coding: utf-8
"""create/download/update :class:`Dataset<supervisely.project.project.Dataset>`"""

# docs
from __future__ import annotations

from typing import Dict, List, Literal, NamedTuple, Optional, Union

from supervisely._utils import abs_url, compress_image_url, is_development
from supervisely.api.module_api import (
    ApiField,
    ModuleApi,
    RemoveableModuleApi,
    UpdateableModule,
)


class DatasetInfo(NamedTuple):
    """ """

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
        # api = sly.Api(server_address="https://app.supervise.ly", token="4r47N...xaTatb")

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
                        reference_image_url='https://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/K/q/jf/...png'),
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
        self, project_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[DatasetInfo]:
        """
        List of Datasets in the given Project.

        :param project_id: Project ID in which the Datasets are located.
        :type project_id: int
        :param filters: List of params to sort output Datasets.
        :type filters: List[dict], optional
        :return: List of all Datasets with information for the given Project. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[DatasetInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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
            #                 reference_image_url="http://app.supervise.ly/z6ut6j8bnaz1vj8aebbgs4-public/images/original/...jpg"),
            #                 DatasetInfo(id=2557,
            #                 name="kiwi",
            #                 description="",
            #                 size="861069",
            #                 project_id=1951,
            #                 images_count=6,
            #                 items_count=6,
            #                 created_at="2021-03-10T09:31:33.701Z",
            #                 updated_at="2021-03-10T09:31:44.196Z",
            #                 reference_image_url="http://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg")
            # ]
        """
        return self.get_list_all_pages(
            "datasets.list",
            {ApiField.PROJECT_ID: project_id, ApiField.FILTER: filters or []},
        )

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_info = api.dataset.get_info_by_id(dataset_id)
        """
        info = self._get_info_by_id(id, "datasets.info")
        if info is None and raise_error is True:
            raise KeyError(f"Dataset with id={id} not found in your account")
        return info

    def create(
        self,
        project_id: int,
        name: str,
        description: Optional[str] = "",
        change_name_if_conflict: Optional[bool] = False,
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
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 116482

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            ds_info = api.dataset.get_list(project_id)
            print(len(ds_info)) # 1

            new_ds = api.dataset.create(project_id, 'new_ds')
            new_ds_info = api.dataset.get_list(project_id)
            print(len(new_ds_info)) # 2
        """
        effective_name = self._get_effective_new_name(
            parent_id=project_id,
            name=name,
            change_name_if_conflict=change_name_if_conflict,
        )
        response = self._api.post(
            "datasets.add",
            {
                ApiField.PROJECT_ID: project_id,
                ApiField.NAME: effective_name,
                ApiField.DESCRIPTION: description,
            },
        )
        return self._convert_json_info(response.json())

    def get_or_create(
        self, project_id: int, name: str, description: Optional[str] = ""
    ) -> DatasetInfo:
        """
        Checks if Dataset with given name already exists in the Project, if not creates Dataset with the given name.

        :param project_id: Project ID in Supervisely.
        :type project_id: int
        :param name: Dataset name.
        :type name: str
        :param description: Dataset description.
        :type description: str, optional
        :return: Information about Dataset. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`DatasetInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 116482

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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
        dataset_info = self.get_info_by_name(project_id, name)
        if dataset_info is None:
            dataset_info = self.create(project_id, name, description=description)
        return dataset_info

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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

        new_datasets = []
        for idx, dataset_id in enumerate(ids):
            dataset = self.get_info_by_id(dataset_id)
            new_dataset_name = dataset.name
            if new_names is not None:
                new_dataset_name = new_names[idx]
            src_images = self._api.image.get_list(dataset.id)
            src_image_ids = [image.id for image in src_images]
            new_dataset = self._api.dataset.create(
                dst_project_id,
                new_dataset_name,
                dataset.description,
                change_name_if_conflict=change_name_if_conflict,
            )
            self._api.image.copy_batch(
                new_dataset.id, src_image_ids, change_name_if_conflict, with_annotations
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
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

    def _convert_json_info(self, info: dict, skip_missing=True):
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.reference_image_url is not None:
            res = res._replace(reference_image_url=res.reference_image_url)
        if res.items_count is None:
            res = res._replace(items_count=res.images_count)
        return DatasetInfo(**res._asdict())

    def remove_permanently(self, ids: Union[int, List]) -> dict:
        """
        Delete permanently datasets with given IDs from the Supervisely server.

        !!! WARNING !!!
        Be careful, this method deletes data from the database, recovery is not possible.

        :param ids: IDs of datasets in Supervisely.
        :type ids: Union[int, List]
        :return: Response content in JSON format
        :rtype: dict
        """
        if isinstance(ids, int):
            datasets = [{ApiField.ID: ids}]
        else:
            datasets = [{ApiField.ID: id} for id in ids]
        response = self._api.post("datasets.remove.permanently", {ApiField.DATASETS: datasets})
        return response.json()

    def get_list_all(
        self,
        filters: Optional[List[Dict[str, str]]] = None,
        sort: Optional[str] = None,
        sort_order: Optional[str] = None,
        per_page: Optional[int] = None,
        page: Union[int, Literal["all"]] = "all",
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
