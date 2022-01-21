# coding: utf-8
"""create/download/update :class:`Project<supervisely.project.project.Project>`"""

from __future__ import annotations
from typing import List, NamedTuple, Dict, Optional
from pandas.core.frame import DataFrame

import pandas as pd
import urllib
from collections import defaultdict

from supervisely_lib.api.module_api import ApiField, CloneableModuleApi, UpdateableModule, RemoveableModuleApi
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.project.project_type import ProjectType
from supervisely_lib.annotation.annotation import TagCollection
from supervisely_lib.task.progress import Progress
import supervisely_lib as sly


class ProjectNotFound(Exception):
    pass


class ExpectedProjectTypeMismatch(Exception):
    pass


class ProjectApi(CloneableModuleApi, UpdateableModule, RemoveableModuleApi):
    """
    API for working with :class:`Project<supervisely_lib.project.project.Project>`. :class:`ProjectApi<ProjectApi>` object is immutable.

    :param api: API connection to the server
    :type api: Api
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # You can connect to API directly
        address = 'https://app.supervise.ly/'
        token = 'Your Supervisely API Token'
        api = sly.Api(address, token)

        # Or you can use API from environment
        os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
        os.environ['API_TOKEN'] = 'Your Supervisely API Token'
        api = sly.Api.from_env()

        project_id = 1951
        project_info = api.project.get_info_by_id(project_id)
    """
    @staticmethod
    def info_sequence():
        """
        NamedTuple ProjectInfo with API Fields containing information about Project.

        :Example:

         .. code-block:: python

            ProjectInfo(id=999,
                        name='Cat_breeds',
                        description='',
                        size='861069',
                        readme='',
                        workspace_id=58,
                        images_count=10,
                        items_count=10,
                        datasets_count=2,
                        created_at='2020-11-17T17:44:28.158Z',
                        updated_at='2021-03-01T10:51:57.545Z',
                        type='images',
                        reference_image_url='http://app.supervise.ly/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg'),
                        custom_data={}
        """
        return [ApiField.ID,
                ApiField.NAME,
                ApiField.DESCRIPTION,
                ApiField.SIZE,
                ApiField.README,
                ApiField.WORKSPACE_ID,
                ApiField.IMAGES_COUNT,  # for compatibility with existing code
                ApiField.ITEMS_COUNT,
                ApiField.DATASETS_COUNT,
                ApiField.CREATED_AT,
                ApiField.UPDATED_AT,
                ApiField.TYPE,
                ApiField.REFERENCE_IMAGE_URL,
                ApiField.CUSTOM_DATA
                ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **ProjectInfo**.
        """
        return 'ProjectInfo'

    def __init__(self, api):
        CloneableModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, workspace_id: int, filters: Optional[List[Dict[str, str]]] = None) -> List[NamedTuple]:
        """
        List of Projects in the given Workspace.

        :param workspace_id: Workspace ID in which the Projects are located.
        :type workspace_id: int
        :param filters: List of params to sort output Projects.
        :type filters: List[dict], optional
        :return: List of all projects with information for the given Workspace. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[NamedTuple]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            workspace_id = 58

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_list = api.project.get_list(workspace_id)
            print(project_list)
            # Output: [
            # ProjectInfo(id=861,
            #             name='Project_COCO',
            #             description='',
            #             size='22172241',
            #             readme='',
            #             workspace_id=58,
            #             images_count=6,
            #             items_count=6,
            #             datasets_count=1,
            #             created_at='2020-11-09T18:21:32.356Z',
            #             updated_at='2020-11-09T18:21:32.356Z',
            #             type='images',
            #             reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg'),
            # ProjectInfo(id=999,
            #             name='Cat_breeds',
            #             description='',
            #             size='861069',
            #             readme='',
            #             workspace_id=58,
            #             images_count=10,
            #             items_count=10,
            #             datasets_count=2,
            #             created_at='2020-11-17T17:44:28.158Z',
            #             updated_at='2021-03-01T10:51:57.545Z',
            #             type='images',
            #             reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg')
            # ]

            # Filtered Project list
            project_list = api.project.get_list(workspace_id, filters=[{ 'field': 'name', 'operator': '=', 'value': 'Cat_breeds'}])
            print(project_list)
            # Output: ProjectInfo(id=999,
            #                     name='Cat_breeds',
            #                     description='',
            #                     size='861069',
            #                     readme='',
            #                     workspace_id=58,
            #                     images_count=10,
            #                     items_count=10,
            #                     datasets_count=2,
            #                     created_at='2020-11-17T17:44:28.158Z',
            #                     updated_at='2021-03-01T10:51:57.545Z',
            #                     type='images',
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg')
            # ]

        """
        return self.get_list_all_pages('projects.list',  {ApiField.WORKSPACE_ID: workspace_id, "filter": filters or []})

    def get_info_by_id(self, id: int, expected_type: Optional[str] = None, raise_error: Optional[bool] = False) -> NamedTuple:
        """
        Get Project information by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :param expected_type: Expected ProjectType.
        :type expected_type: ProjectType, optional
        :param raise_error: If True raise error if given name is missing in the Project, otherwise skips missing names.
        :type raise_error: bool, optional
        :raises: Error if type of project is not None and != expected type
        :return: Information about Project. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_info = api.project.get_info_by_id(project_id)
            print(project_info)
            # Output: ProjectInfo(id=861,
            #                     name='fruits_annotated',
            #                     description='',
            #                     size='22172241',
            #                     readme='',
            #                     workspace_id=58,
            #                     images_count=6,
            #                     items_count=6,
            #                     datasets_count=1,
            #                     created_at='2020-11-09T18:21:32.356Z',
            #                     updated_at='2020-11-09T18:21:32.356Z',
            #                     type='images',
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg')

        """
        info = self._get_info_by_id(id, 'projects.info')
        self._check_project_info(info, id=id, expected_type=expected_type, raise_error=raise_error)
        return info

    def get_info_by_name(self, parent_id: int, name: str, expected_type: Optional[ProjectType] = None,
                         raise_error: Optional[bool] = False) -> NamedTuple:
        """
        Get Project information by name.

        :param parent_id: Workspace ID.
        :type parent_id: int
        :param name: Project name.
        :type name: str
        :param expected_type: Expected ProjectType.
        :type expected_type: ProjectType, optional
        :param raise_error: If True raise error if given name is missing in the Project, otherwise skips missing names.
        :type raise_error: bool, optional
        :return: Information about Project. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_info = api.project.get_info_by_name(58, "fruits_annotated")
            print(project_info)
            # Output: ProjectInfo(id=861,
            #                     name='fruits_annotated',
            #                     description='',
            #                     size='22172241',
            #                     readme='',
            #                     workspace_id=58,
            #                     images_count=6,
            #                     items_count=6,
            #                     datasets_count=1,
            #                     created_at='2020-11-09T18:21:32.356Z',
            #                     updated_at='2020-11-09T18:21:32.356Z',
            #                     type='images',
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg')
        """
        info = super().get_info_by_name(parent_id, name)
        self._check_project_info(info, name=name, expected_type=expected_type, raise_error=raise_error)
        return info

    def _check_project_info(self, info, id: Optional[int]=None, name: Optional[str]=None, expected_type=None, raise_error=False):
        """
        Checks if a project exists with a given id and type of project == expected type
        :param info: project metadata information
        :param id: int
        :param name: str
        :param expected_type: type of data we expext to get info
        :param raise_error: bool
        """
        if raise_error is False:
            return

        str_id = ""
        if id is not None:
            str_id += "id: {!r} ".format(id)
        if name is not None:
            str_id += "name: {!r}".format(name)

        if info is None:
            raise ProjectNotFound("Project {} not found".format(str_id))
        if expected_type is not None and info.type != str(expected_type):
            raise ExpectedProjectTypeMismatch("Project {!r} has type {!r}, but expected type is {!r}"
                                              .format(str_id, info.type, expected_type))

    def get_meta(self, id: int) -> Dict:
        """
        Get ProjectMeta by Project ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: ProjectMeta dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_meta = api.project.get_meta(project_id)
            print(project_meta)
            # Output: {
            #     "classes":[
            #         {
            #             "id":22310,
            #             "title":"kiwi",
            #             "shape":"bitmap",
            #             "hotkey":"",
            #             "color":"#FF0000"
            #         },
            #         {
            #             "id":22309,
            #             "title":"lemon",
            #             "shape":"bitmap",
            #             "hotkey":"",
            #             "color":"#51C6AA"
            #         }
            #     ],
            #     "tags":[],
            #     "projectType":"images"
            # }
        """
        response = self._api.post('projects.meta', {'id': id})
        return response.json()

    def create(self, workspace_id: int, name: str, type: ProjectType = ProjectType.IMAGES, description: Optional[str] = "",
               change_name_if_conflict: Optional[bool] = False) -> NamedTuple:
        """
        Create Project with given name in the given Workspace ID.

        :param workspace_id: Workspace ID in Supervisely where Project will be created.
        :type workspace_id: int
        :param name: Project Name.
        :type name: str
        :param type: Type of created Project.
        :type type: ProjectType
        :param description: Project description.
        :type description: str
        :param change_name_if_conflict: Checks if given name already exists and adds suffix to the end of the name.
        :type change_name_if_conflict: bool, optional
        :return: Information about Project. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`NamedTuple`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            workspace_id = 8

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_proj = api.project.create(workspace_id, "fruits_test", sly.ProjectType.IMAGES)
            print(new_proj)
            # Output: ProjectInfo(id=1993,
            #                     name='fruits_test',
            #                     description='',
            #                     size='0',
            #                     readme='',
            #                     workspace_id=58,
            #                     images_count=None,
            #                     items_count=None,
            #                     datasets_count=None,
            #                     created_at='2021-03-11T09:28:42.585Z',
            #                     updated_at='2021-03-11T09:28:42.585Z',
            #                     type='images',
            #                     reference_image_url=None)
        """
        effective_name = self._get_effective_new_name(
            parent_id=workspace_id, name=name, change_name_if_conflict=change_name_if_conflict)
        response = self._api.post('projects.add', {ApiField.WORKSPACE_ID: workspace_id,
                                                   ApiField.NAME: effective_name,
                                                   ApiField.DESCRIPTION: description,
                                                   ApiField.TYPE: str(type)})
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        return 'projects.editInfo'

    def update_meta(self, id: int, meta: Dict) -> None:
        """
        Updates given Project with given ProjectMeta.

        :param id: Project ID in Supervisely.
        :type id: int
        :param meta: ProjectMeta dict
        :type meta: dict
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            lemons_proj_id = 1951
            kiwis_proj_id = 1952

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_meta = api.project.get_meta(lemons_proj_id)
            updated_meta = api.project.update_meta(kiwis_proj_id, project_meta)
        """
        self._api.post('projects.meta.update', {ApiField.ID: id, ApiField.META: meta})

    def _clone_api_method_name(self):
        return 'projects.clone'

    def get_datasets_count(self, id: int) -> int:
        """
        Number of Datasets in the given Project by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: Number of Datasets in the given Project
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 454

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_ds_count = api.project.get_datasets_count(project_id)
            print(project_ds_count)
            # Output: 4
        """
        datasets = self._api.dataset.get_list(id)
        return len(datasets)

    def get_images_count(self, id: int) -> int:
        """
        Number of images in the given Project by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: Number of images in the given Project
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 454

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_imgs_count = api.project.get_images_count(project_id)
            print(project_imgs_count)
            # Output: 24
        """
        datasets = self._api.dataset.get_list(id)
        return sum([dataset.images_count for dataset in datasets])

    def _remove_api_method_name(self):
        return 'projects.remove'

    def merge_metas(self, src_project_id: int, dst_project_id: int) -> Dict:
        """
        Merges ProjectMeta from given Project to given destination Project.

        :param src_project_id: Source Project ID.
        :type src_project_id: int
        :param dst_project_id: Destination Project ID.
        :type dst_project_id: int
        :return: ProjectMeta dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            lemons_proj_id = 1951
            kiwis_proj_id = 1980

            merged_projects = api.project.merge_metas(lemons_proj_id, kiwis_proj_id)
        """
        if src_project_id == dst_project_id:
            return self.get_meta(src_project_id)

        src_meta = ProjectMeta.from_json(self.get_meta(src_project_id))
        dst_meta = ProjectMeta.from_json(self.get_meta(dst_project_id))

        new_dst_meta = src_meta.merge(dst_meta)
        new_dst_meta_json = new_dst_meta.to_json()
        self.update_meta(dst_project_id, new_dst_meta.to_json())

        return new_dst_meta_json

    def get_activity(self, id: int, progress_cb: Optional[Progress]=None) -> DataFrame:
        """
        Get Project activity by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: `Pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        :rtype: :class:`DataFrame`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_activity = api.project.get_activity(project_id)
            print(project_activity)
            # Output:    userId               action  ... tagId             meta
            #         0       7  annotation_duration  ...  None  {'duration': 1}
            #         1       7  annotation_duration  ...  None  {'duration': 2}
            #         2       7        create_figure  ...  None               {}
            #
            #         [3 rows x 18 columns]
        """
        proj_info = self.get_info_by_id(id)
        workspace_info = self._api.workspace.get_info_by_id(proj_info.workspace_id)
        activity = self._api.team.get_activity(workspace_info.team_id, filter_project_id=id, progress_cb=progress_cb)
        df = pd.DataFrame(activity)
        return df

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.reference_image_url is not None:
            res = res._replace(reference_image_url=urllib.parse.urljoin(self._api.server_address, res.reference_image_url))
        if res.items_count is None:
            res = res._replace(items_count=res.images_count)
        return res

    def get_stats(self, id: int) -> Dict:
        """
        Get Project stats by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: Project statistics
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_stats = api.project.get_stats(project_id)
        """
        response = self._api.post('projects.stats', {ApiField.ID: id})
        return response.json()

    def url(self, id: int) -> str:
        """
        Get Project URL by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: Project URL
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_url = api.project.url(project_id)
            print(project_url)
            # Output: http://supervise.ly/projects/1951/datasets
        """
        result = urllib.parse.urljoin(self._api.server_address, 'projects/{}/datasets'.format(id))
        return result

    def update_custom_data(self, id: int, data: Dict) -> Dict:
        """
        Updates custom data of the Project by ID

        :param id: Project ID in Supervisely.
        :type id: int
        :param data: Custom data
        :type data: dict
        :return: Project information in dict format
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951
            custom_data = {1:2}

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            new_info = api.project.update_custom_data(project_id, custom_data)
        """
        if type(data) is not dict:
            raise TypeError('Meta must be dict, not {!r}'.format(type(data)))
        response = self._api.post('projects.editInfo', {ApiField.ID: id, ApiField.CUSTOM_DATA: data})
        return response.json()

    def download_images_tags(self, id: int, progress_cb: Optional[Progress]=None) -> defaultdict:
        """
        Get matching tag names to ImageInfos.

        :param id: Project ID in Supervisely.
        :type id: int
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: Progress, optional
        :return: Defaultdict matching tag names to ImageInfos
        :rtype: :class:`defaultdict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_id = 8200
            tags_to_infos = api.project.download_images_tags(project_id)
            for tag_name in tags_to_infos:
                print(tag_name, tags_to_infos[tag_name])
            # Output:
            # train [ImageInfo(id=2389064, name='IMG_4451_JjH4WPkHlk.jpeg', link=None, hash='6EpjCL+lBdMBYo...
            # val [ImageInfo(id=2389066, name='IMG_1836.jpeg', link=None, hash='Si0WvJreU6pmrx1EDa1itkqqSkQkZFzNJSu...
        """
        # returns dict: tagname->images infos
        #project_meta = self.get_meta(id) #TODO alex check bug
        meta = self.get_meta(id) #TODO alex check bug
        project_meta = sly.ProjectMeta.from_json(meta) #TODO alex check bug
        id_to_tagmeta = project_meta.tag_metas.get_id_mapping()
        tag2images = defaultdict(list)
        for dataset in self._api.dataset.get_list(id):
            ds_images = self._api.image.get_list(dataset.id)
            for img_info in ds_images:
                tags = TagCollection.from_api_response(img_info.tags, project_meta.tag_metas, id_to_tagmeta)
                for tag in tags:
                    tag2images[tag.name].append(img_info)
                if progress_cb is not None:
                    progress_cb(1)
        return tag2images
