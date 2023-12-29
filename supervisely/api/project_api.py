# coding: utf-8
"""create/download/update :class:`Project<supervisely.project.project.Project>`"""

# docs
from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Union,
)

from tqdm import tqdm

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

from datetime import datetime, timedelta

from supervisely import logger
from supervisely._utils import abs_url, compress_image_url, is_development
from supervisely.annotation.annotation import TagCollection
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.module_api import (
    ApiField,
    CloneableModuleApi,
    RemoveableModuleApi,
    UpdateableModule,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_meta import ProjectMetaJsonFields as MetaJsonF
from supervisely.project.project_settings import ProjectSettings
from supervisely.project.project_type import (
    _MULTISPECTRAL_TAG_NAME,
    _MULTIVIEW_TAG_NAME,
    ProjectType,
)


class ProjectNotFound(Exception):
    """ """

    pass


class ExpectedProjectTypeMismatch(Exception):
    """ """

    pass


class ProjectInfo(NamedTuple):
    """ """

    id: int
    name: str
    description: str
    size: int
    readme: str
    workspace_id: int
    images_count: int  # for compatibility with existing code
    items_count: int
    datasets_count: int
    created_at: str
    updated_at: str
    type: str
    reference_image_url: str
    custom_data: dict
    backup_archive: dict
    team_id: int
    settings: dict

    @property
    def image_preview_url(self):
        if self.type in [str(ProjectType.POINT_CLOUDS), str(ProjectType.POINT_CLOUD_EPISODES)]:
            res = "https://user-images.githubusercontent.com/12828725/199022135-4161917c-05f8-4681-9dc1-b5e10ee8bb0f.png"
        else:
            res = self.reference_image_url
            if is_development():
                res = abs_url(res)
            res = compress_image_url(url=res, height=200)
        return res

    @property
    def url(self):
        res = f"projects/{self.id}/datasets"
        if is_development():
            res = abs_url(res)
        return res


class ProjectApi(CloneableModuleApi, UpdateableModule, RemoveableModuleApi):
    """
    API for working with :class:`Project<supervisely.project.project.Project>`. :class:`ProjectApi<ProjectApi>` object is immutable.

    :param api: API connection to the server
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
                        custom_data={},
                        backup_archive={},
                        team_id=2
        """
        return [
            ApiField.ID,
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
            ApiField.CUSTOM_DATA,
            ApiField.BACKUP_ARCHIVE,
            ApiField.TEAM_ID,
            ApiField.SETTINGS,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **ProjectInfo**.
        """
        return "ProjectInfo"

    def __init__(self, api):
        CloneableModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(
        self, workspace_id: int, filters: Optional[List[Dict[str, str]]] = None
    ) -> List[ProjectInfo]:
        """
        List of Projects in the given Workspace.

        :param workspace_id: Workspace ID in which the Projects are located.
        :type workspace_id: int
        :param filters: List of params to sort output Projects.
        :type filters: List[dict], optional
        :return: List of all projects with information for the given Workspace. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ProjectInfo]`
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
            #             reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg',
            #             custom_data={},
            #             backup_archive={}),
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
            #             reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg'),
            #             custom_data={},
            #             backup_archive={})
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
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg'),
            #                     custom_data={},
            #                     backup_archive={})
            # ]

        """
        return self.get_list_all_pages(
            "projects.list",
            {ApiField.WORKSPACE_ID: workspace_id, "filter": filters or []},
        )

    def get_info_by_id(
        self,
        id: int,
        expected_type: Optional[str] = None,
        raise_error: Optional[bool] = False,
    ) -> ProjectInfo:
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
        :rtype: :class:`ProjectInfo`
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
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg'),
            #                     custom_data={},
            #                     backup_archive={})


        """
        info = self._get_info_by_id(id, "projects.info")
        self._check_project_info(info, id=id, expected_type=expected_type, raise_error=raise_error)
        return info

    def get_info_by_name(
        self,
        parent_id: int,
        name: str,
        expected_type: Optional[ProjectType] = None,
        raise_error: Optional[bool] = False,
    ) -> ProjectInfo:
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
        :rtype: :class:`ProjectInfo`
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
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg'),
            #                     custom_data={},
            #                     backup_archive={})
        """
        info = super().get_info_by_name(parent_id, name)
        self._check_project_info(
            info, name=name, expected_type=expected_type, raise_error=raise_error
        )
        return info

    def _check_project_info(
        self,
        info,
        id: Optional[int] = None,
        name: Optional[str] = None,
        expected_type=None,
        raise_error=False,
    ):
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
            raise ExpectedProjectTypeMismatch(
                "Project {!r} has type {!r}, but expected type is {!r}".format(
                    str_id, info.type, expected_type
                )
            )

    def get_meta(self, id: int, with_settings: bool = False) -> Dict:
        """
        Get ProjectMeta by Project ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :param with_settings: Add settings field to the meta. By default False.
        :type with_settings: bool

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
        json_response = self._api.post("projects.meta", {"id": id}).json()

        if with_settings is True:
            json_settings = self.get_settings(id)

            if json_settings.get("groupImagesByTagId") is not None:
                for tag in json_response["tags"]:
                    if tag["id"] == json_settings["groupImagesByTagId"]:
                        json_response[MetaJsonF.PROJECT_SETTINGS] = ProjectSettings(
                            multiview_enabled=json_settings["groupImages"],
                            multiview_tag_name=tag["name"],
                            multiview_tag_id=json_settings["groupImagesByTagId"],
                            multiview_is_synced=json_settings["groupImagesSync"],
                        ).to_json()
                        break

        return json_response

    def clone_advanced(
        self,
        id,
        dst_workspace_id,
        dst_name,
        with_meta=True,
        with_datasets=True,
        with_items=True,
        with_annotations=True,
    ):
        """ """
        if not with_meta and with_annotations:
            raise ValueError(
                "with_meta parameter must be True if with_annotations parameter is True"
            )
        if not with_datasets and with_items:
            raise ValueError("with_datasets parameter must be True if with_items parameter is True")
        response = self._api.post(
            self._clone_api_method_name(),
            {
                ApiField.ID: id,
                ApiField.WORKSPACE_ID: dst_workspace_id,
                ApiField.NAME: dst_name,
                ApiField.INCLUDE: {
                    ApiField.CLASSES: with_meta,
                    ApiField.PROJECT_TAGS: with_meta,
                    ApiField.DATASETS: with_datasets,
                    ApiField.IMAGES: with_items,
                    ApiField.IMAGES_TAGS: with_items,
                    ApiField.ANNOTATION_OBJECTS: with_annotations,
                    ApiField.ANNOTATION_OBJECTS_TAGS: with_annotations,
                    ApiField.FIGURES: with_annotations,
                    ApiField.FIGURES_TAGS: with_annotations,
                },
            },
        )
        return response.json()[ApiField.TASK_ID]

    def create(
        self,
        workspace_id: int,
        name: str,
        type: ProjectType = ProjectType.IMAGES,
        description: Optional[str] = "",
        change_name_if_conflict: Optional[bool] = False,
    ) -> ProjectInfo:
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
        :rtype: :class:`ProjectInfo`
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
            #                     reference_image_url=None),
            #                     custom_data={},
            #                     backup_archive={})

        """
        effective_name = self._get_effective_new_name(
            parent_id=workspace_id,
            name=name,
            change_name_if_conflict=change_name_if_conflict,
        )
        response = self._api.post(
            "projects.add",
            {
                ApiField.WORKSPACE_ID: workspace_id,
                ApiField.NAME: effective_name,
                ApiField.DESCRIPTION: description,
                ApiField.TYPE: str(type),
            },
        )
        return self._convert_json_info(response.json())

    def _get_update_method(self):
        """ """
        return "projects.editInfo"

    def update_meta(self, id: int, meta: Union[Dict, ProjectMeta]) -> None:
        """
        Updates given Project with given ProjectMeta.

        :param id: Project ID in Supervisely.
        :type id: int
        :param meta: ProjectMeta object or ProjectMeta in JSON format.
        :type meta: :class:`ProjectMeta` or dict

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

            # Using ProjectMeta in JSON format

            project_meta_json = api.project.get_meta(lemons_proj_id)
            api.project.update_meta(kiwis_proj_id, project_meta_json)

            # Using ProjectMeta object

            project_meta_json = api.project.get_meta(lemons_proj_id)
            project_meta = sly.ProjectMeta.from_json(path_to_meta)
            api.project.update_meta(kiwis_proj_id, project_meta)

            # Using programmatically created ProjectMeta

            cat_class = sly.ObjClass("cat", sly.Rectangle, color=[0, 255, 0])
            scene_tag = sly.TagMeta("scene", sly.TagValueType.ANY_STRING)
            project_meta = sly.ProjectMeta(obj_classes=[cat_class], tag_metas=[scene_tag])
            api.project.update_meta(kiwis_proj_id, project_meta)

            # Update ProjectMeta from local `meta.json`
            from supervisely.io.json import load_json_file

            path_to_meta = "/path/project/meta.json"
            project_meta_json = load_json_file(path_to_meta)
            api.project.update_meta(kiwis_proj_id, project_meta)
        """

        m = meta
        if isinstance(meta, dict):
            m = ProjectMeta.from_json(meta)

        if m.project_settings is not None:
            group_tag = m.get_tag_meta(m.project_settings.multiview_tag_name)
            if group_tag is None:
                raise RuntimeError("dsffsdf")

        self._api.post("projects.meta.update", {ApiField.ID: id, ApiField.META: m.to_json()})

        if m.project_settings is not None:
            new_m = ProjectMeta.from_json(self.get_meta(id))
            group_tag = new_m.get_tag_meta(m.project_settings.multiview_tag_name)
            new_m.project_settings = m.project_settings.clone(multiview_tag_id=group_tag.sly_id)
            self.update_settings(id, new_m.project_settings)

        return new_m

        meta_json = None
        if isinstance(meta, ProjectMeta):
            meta_json = meta.to_json()
        else:
            meta_json = meta
            meta = ProjectMeta.from_json(meta_json)

        self._api.post("projects.meta.update", {ApiField.ID: id, ApiField.META: meta_json})

        if meta.project_settings is not None:
            s = meta.project_settings
            new_settings = {
                "groupImages": s.multiview_enabled,
                "groupImagesByTagId": s.multiview_tag_id,
                "groupImagesSync": s.multiview_is_synced,
            }

            if s.multiview_enabled is True:
                add_lacking_tag = False
                for tag in self.get_meta(id)["tags"]:
                    if s.multiview_tag_name is None and s.multiview_tag_id is None:
                        logger.warn(
                            f"Oops! It seems like you have enabled the multi-view mode in meta.json, but forgotten to specify a tag. Adding it for you..."
                        )
                        add_lacking_tag = True

                    if (
                        add_lacking_tag is True
                        or tag["name"] == s.multiview_tag_name
                        or tag["id"] == s.multiview_tag_id
                    ):
                        logger.info(f"Multi-view mode has been enabled with '{tag['name']}' tag.")
                        new_settings["groupImagesByTagId"] = tag["id"]

                        break

            self.update_settings(id, new_settings)

    def _clone_api_method_name(self):
        """ """
        return "projects.clone"

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
        """"""
        return "projects.remove"

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

    def get_activity(
        self, id: int, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> DataFrame:
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
        import pandas as pd

        proj_info = self.get_info_by_id(id)
        workspace_info = self._api.workspace.get_info_by_id(proj_info.workspace_id)
        activity = self._api.team.get_activity(
            workspace_info.team_id, filter_project_id=id, progress_cb=progress_cb
        )
        df = pd.DataFrame(activity)
        return df

    def _convert_json_info(self, info: dict, skip_missing=True) -> ProjectInfo:
        """ """
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.reference_image_url is not None:
            res = res._replace(reference_image_url=res.reference_image_url)
        if res.items_count is None:
            res = res._replace(items_count=res.images_count)
        return ProjectInfo(**res._asdict())

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
        response = self._api.post("projects.stats", {ApiField.ID: id})
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
        res = f"projects/{id}/datasets"
        if is_development():
            res = abs_url(res)
        return res

    def update_custom_data(self, id: int, data: Dict, silent: bool = False) -> Dict:
        """
        Updates custom data of the Project by ID

        :param id: Project ID in Supervisely.
        :type id: int
        :param data: Custom data
        :type data: dict
        :param silent: determines whether the `updatedAt` timestamp should be updated or not, if False - update `updatedAt`
        :type silent: bool
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
            raise TypeError("Meta must be dict, not {!r}".format(type(data)))
        response = self._api.post(
            "projects.editInfo",
            {ApiField.ID: id, ApiField.CUSTOM_DATA: data, ApiField.SILENT: silent},
        )
        return response.json()

    def get_settings(self, id: int) -> Dict[str, str]:
        info = self._get_info_by_id(id, "projects.info")
        return info.settings

    def update_settings(self, id: int, settings: Dict[str, str]) -> None:
        """
        Updates project wuth given project settings by id.

        :param id: Project ID
        :type id: int
        :param settings: Project settings
        :type settings: Dict[str, str]
        """
        self._api.post("projects.settings.update", {ApiField.ID: id, ApiField.SETTINGS: settings})

    def download_images_tags(
        self, id: int, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> defaultdict:
        """
        Get matching tag names to ImageInfos.

        :param id: Project ID in Supervisely.
        :type id: int
        :param progress_cb: Function for tracking download progress.
        :type progress_cb: tqdm or callable, optional
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
        project_meta = self.get_meta(id)
        id_to_tagmeta = project_meta.tag_metas.get_id_mapping()
        tag2images = defaultdict(list)
        for dataset in self._api.dataset.get_list(id):
            ds_images = self._api.image.get_list(dataset.id)
            for img_info in ds_images:
                tags = TagCollection.from_api_response(
                    img_info.tags, project_meta.tag_metas, id_to_tagmeta
                )
                for tag in tags:
                    tag2images[tag.name].append(img_info)
                if progress_cb is not None:
                    progress_cb(1)
        return tag2images

    def images_grouping(self, id: int, enable: bool, tag_name: str, sync: bool = False) -> None:
        """Enables and disables images grouping by given tag name.

        :param id: Project ID, where images grouping will be enabled
        :type id: int
        :param enable: if True groups images by given tag name, otherwise disables images grouping
        :type enable: bool
        :param tag_name: Name of the tag. Images will be grouped by this tag
        :type tag_name: str
        """
        project_meta_json = self.get_meta(id)
        project_meta = ProjectMeta.from_json(project_meta_json)
        group_tag_meta = project_meta.get_tag_meta(tag_name)
        if group_tag_meta is None:
            raise Exception(f"Tag {tag_name} doesn't exists in the given project")

        group_tag_id = group_tag_meta.sly_id
        project_settings = {
            "groupImages": enable,
            "groupImagesByTagId": group_tag_id,
            "groupImagesSync": sync,
        }
        self.update_settings(id=id, settings=project_settings)

    def get_or_create(
        self,
        workspace_id: int,
        name: str,
        type: Optional[str] = ProjectType.IMAGES,
        description: Optional[str] = "",
    ) -> ProjectInfo:
        """Returns project info if project with given name exists in given workspace, otherwise creates new project
        and returns info about it.

        :param workspace_id: Workspace ID in which the Project will be searched or created.
        :type workspace_id: int
        :param name: name of the project to search or create
        :type name: str
        :param type: type of the project to create
        :type type: Optional[str], default ProjectType.IMAGES
        :param description: description of the project to create
        :type description: Optional[str]
        :return: ProjectInfo about found or created project
        :rtype: ProjectInfo
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            project_name = "my_project"
            workspace_id = 123
            project_info = api.project.get_or_create(workspace_id, project_name)
        """
        info = self.get_info_by_name(workspace_id, name)
        if info is None:
            info = self.create(workspace_id, name, type=type, description=description)
        return info

    def edit_info(
        self,
        id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        readme: Optional[str] = None,
        custom_data: Optional[Dict[Any, Any]] = None,
        project_type: Optional[str] = None,
    ) -> ProjectInfo:
        """Edits the project info by given parameters.

        :param id: ID of the project to edit info
        :type id: int
        :param name: new name of the project
        :type name: Optional[str]
        :param description: new description of the project
        :type description: Optional[str]
        :param readme: new readme of the project
        :type readme: Optional[str]
        :param custom_data: new custom data of the project
        :type custom_data: Optional[Dict[Any, Any]]
        :param project_type: new type of the project
        :type project_type: Optional[str]
        :return: ProjectInfo of the edited project
        :rtype: ProjectInfo
        :raises ValueError: if no arguments are specified
        :raises ValueError: if invalid project type is specified
        :raises ValueError: if project with given id already has given type
        :raises ValueError: if conversion from current project type to given project type is not supported
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            project_id = 123
            new_name = "new_name"
            new_description = "new_description"
            project_info = api.project.edit_info(project_id, name=new_name, description=new_description)
        """
        if (
            name is None
            and description is None
            and readme is None
            and custom_data is None
            and project_type is None
        ):
            raise ValueError("one of the arguments has to be specified")

        body = {ApiField.ID: id}
        if name is not None:
            body[ApiField.NAME] = name
        if description is not None:
            body[ApiField.DESCRIPTION] = description
        if readme is not None:
            body[ApiField.README] = readme
        if custom_data is not None:
            body[ApiField.CUSTOM_DATA] = custom_data
        if project_type is not None:
            if isinstance(project_type, ProjectType):
                project_type = str(project_type)
            if project_type not in ProjectType.values():
                raise ValueError(f"project type must be one of: {ProjectType.values()}")
            project_info = self.get_info_by_id(id)
            current_project_type = project_info.type
            if project_type == current_project_type:
                raise ValueError(f"project with id {id} already has type {project_type}")
            if not (
                current_project_type == str(ProjectType.POINT_CLOUDS)
                and project_type == str(ProjectType.POINT_CLOUD_EPISODES)
            ):
                raise ValueError(
                    f"conversion from {current_project_type} to {project_type} is not supported "
                )
            body[ApiField.TYPE] = project_type

        response = self._api.post(self._get_update_method(), body)
        return self._convert_json_info(response.json())

    def pull_meta_ids(self, id: int, meta: ProjectMeta) -> None:
        """Updates given ProjectMeta with ids from server.

        :param id: Project ID
        :type id: int
        :param meta: ProjectMeta to update ids
        :type meta: ProjectMeta
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            project_id = 123
            # We already have ProjectMeta and now we want to update ids in it
            # from server
            meta: sly.ProjectMeta

            api.project.pull_meta_ids(project_id, meta)
        """
        # to update ids in existing project meta
        meta_json = self.get_meta(id)
        server_meta = ProjectMeta.from_json(meta_json)
        meta.obj_classes.refresh_ids_from(server_meta.obj_classes)
        meta.tag_metas.refresh_ids_from(server_meta.tag_metas)

    def move(self, id: int, workspace_id: int) -> None:
        """
        Move project between workspaces within current team.

        :param id: Project ID
        :type id: int
        :param workspace_id: Workspace ID the project will move in
        :type workspace_id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            workspace_id = 688
            project_id = 17173

            api.project.move(id=project_id, workspace_id=workspace_id)
        """
        self._api.post(
            "projects.workspace.set", {ApiField.ID: id, ApiField.WORKSPACE_ID: workspace_id}
        )

    def archive_batch(
        self, ids: List[int], archive_urls: List[str], ann_archive_urls: Optional[List[str]] = None
    ) -> None:
        """
        Archive Projects by ID and save backup URLs in Project info for every Project.

        :param ids: Project IDs in Supervisely.
        :type ids: List[int]
        :param archive_urls: Shared URLs of files backup on Dropbox.
        :type archive_urls: List[str]
        :param ann_archive_urls: Shared URLs of annotations backup on Dropbox.
        :type ann_archive_urls: List[str], optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            ids = [18464, 18461]
            archive_urls = ['https://www.dropbox.com/...', 'https://www.dropbox.com/...']

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.project.archive_batch(ids, archive_urls, ann_archive_urls)
        """
        if len(ids) != len(archive_urls):
            raise ValueError(
                "The list with Project IDs must have the same length as the list with URLs for archives"
            )
        for id, archive_url, ann_archive_url in zip(
            ids, archive_urls, ann_archive_urls or [None] * len(ids)
        ):
            request_params = {
                ApiField.ID: id,
                ApiField.ARCHIVE_URL: archive_url,
            }
            if ann_archive_url is not None:
                request_params[ApiField.ANN_ARCHIVE_URL] = ann_archive_url

            self._api.post("projects.remove.permanently", {ApiField.PROJECTS: [request_params]})

    def archive(self, id: int, archive_url: str, ann_archive_url: Optional[str] = None) -> None:
        """
        Archive Project by ID and save backup URLs in Project info.

        :param id: Project ID in Supervisely.
        :type id: int
        :param archive_url: Shared URL of files backup on Dropbox.
        :type archive_url: str
        :param ann_archive_url: Shared URL of annotations backup on Dropbox.
        :type ann_archive_url: str, optional
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            id = 18464
            archive_url = 'https://www.dropbox.com/...'

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.project.archive(id, archive_url, ann_archive_url)
        """
        if ann_archive_url is None:
            self.archive_batch([id], [archive_url])
        else:
            self.archive_batch([id], [archive_url], [ann_archive_url])

    def get_archivation_list(
        self,
        to_day: Optional[int] = None,
        from_day: Optional[int] = None,
        skip_exported: Optional[bool] = None,
        sort: Optional[
            Literal[
                "id",
                "title",
                "size",
                "createdAt",
                "updatedAt",
            ]
        ] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
        account_type: Optional[str] = None,
    ) -> List[ProjectInfo]:
        """
        List of all projects in all available workspaces that can be archived.

        :param to_day: Sets the number of days from today. If the project has not been updated during this period, it will be added to the list.
        :type to_day: int, optional
        :param from_day: Sets the number of days from today. If the project has not been updated before this period, it will be added to the list.
        :type from_day: int, optional
        :param skip_exported: Determines whether to skip already archived projects.
        :type skip_exported: bool, optional.
        :param sort: Specifies by which parameter to sort the project list.
        :type sort: Optional[Literal["id", "title", "size", "createdAt", "updatedAt"]]
        :param sort_order: Determines which value to list from.
        :type sort_order: Optional[Literal["asc", "desc"]]
        :return: List of all projects with information. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[ProjectInfo]`
        :Usage example:

        .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_list = api.project.get_archivation_list()
            print(project_list)
            # Output: [
            # ProjectInfo(id=861,
            #             name='Project_COCO'
            #             size='22172241',
            #             workspace_id=58,
            #             created_at='2020-11-09T18:21:32.356Z',
            #             updated_at='2020-11-09T18:21:32.356Z',
            #             type='images',),
            # ProjectInfo(id=777,
            #             name='Trucks',
            #             size='76154769',
            #             workspace_id=58,
            #             created_at='2021-07-077T17:44:28.158Z',
            #             updated_at='2023-07-15T12:33:45.747Z',
            #             type='images',)
            # ]

            # Project list for desired date range
            project_list = api.project.get_archivation_list(to_day=2)
            print(project_list)
            # Output: ProjectInfo(id=777,
            #                     name='Trucks',
            #                     size='76154769',
            #                     workspace_id=58,
            #                     created_at='2021-07-077T17:44:28.158Z',
            #                     updated_at='2023-07-15T12:33:45.747Z',
            #                     type='images',)
            # ]

        """
        kwargs = {}

        filters = []
        if from_day is not None:
            date = (datetime.utcnow() - timedelta(days=from_day)).strftime("%Y-%m-%dT%H:%M:%SZ")
            filer_from = {
                ApiField.FIELD: ApiField.UPDATED_AT,
                "operator": ">=",
                ApiField.VALUE: date,
            }
            filters.append(filer_from)
        if to_day is not None:
            date = (datetime.utcnow() - timedelta(days=to_day)).strftime("%Y-%m-%dT%H:%M:%SZ")
            filer_to = filer_from = {
                ApiField.FIELD: ApiField.UPDATED_AT,
                "operator": "<=",
                ApiField.VALUE: date,
            }
            filters.append(filer_to)
        if len(filters) != 0:
            kwargs["filters"] = filters

        if skip_exported is None:
            kwargs["skip_exported"] = skip_exported

        if sort is not None:
            kwargs["sort"] = sort

        if sort_order is not None:
            kwargs["sort_order"] = sort_order

        if account_type is not None:
            kwargs["account_type"] = account_type

        response = self.get_list_all(**kwargs)

        return response.get("entities")

    def check_imageset_backup(self, id: int) -> Optional[Dict]:
        """
        Check if a backup of the project image set exists. If yes, it returns a link to the archive.

        :param id: Project ID
        :type id: int
        :return: dict with shared URL of files backup or None
        :rtype: Dict, optional
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

            response = check_imageset_backup(project_id)
            archive_url = response['imagesArchiveUrl']

        """
        response = self._api.get("projects.images.get-backup-archive", {ApiField.ID: id})

        return response.json()

    def append_classes(self, id: int, classes: Union[List[ObjClass], ObjClassCollection]) -> None:
        """
        Append new classes to given Project.

        :param id: Project ID in Supervisely.
        :type id: int
        :param classes: New classes
        :type classes: :class: ObjClassCollection or List[ObjClass]
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            proj_id = 28145
            lung_obj_class = sly.ObjClass("lung", sly.Mask3D)
            api.project.append_classes(proj_id, [lung_obj_class])
        """
        meta_json = self.get_meta(id)
        meta = ProjectMeta.from_json(meta_json)
        meta = meta.add_obj_classes(classes)
        self.update_meta(id, meta)

    def _set_custom_grouping_settings(self, id: int, group_images: bool, tag_name: str, sync: bool):
        """Sets the project settings for custom grouping.

        :param id: Project ID to set custom grouping settings.
        :type id: int
        :param group_images: if True enables images grouping by tag
        :type group_images: bool
        :param tag_name: Name of the tag. Images will be grouped by this tag
        :type tag_name: str
        :param sync: if True images will have synchronized view and labeling
        :type sync: bool
        """
        group_tag_meta = TagMeta(tag_name, TagValueType.ANY_STRING)
        project_meta = ProjectMeta.from_json(self.get_meta(id))
        project_meta = project_meta.add_tag_meta(group_tag_meta)
        self.update_meta(id, project_meta)
        self.images_grouping(id, enable=group_images, tag_name=tag_name, sync=sync)

    def set_multispectral_settings(self, project_id: int) -> None:
        """Sets the project settings for multispectral images.
        Images will be grouped by tag and have synchronized view and labeling.

        :param project_id: Project ID to set multispectral settings.
        :type project_id: int
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            api.project.set_multispectral_settings(project_id=123)
        """

        self._set_custom_grouping_settings(
            id=project_id,
            group_images=True,
            tag_name=_MULTISPECTRAL_TAG_NAME,
            sync=True,
        )

    def set_multiview_settings(self, project_id: int) -> None:
        """Sets the project settings for multiview images.
        Images will be grouped by tag and have synchronized view and labeling.

        :param project_id: Project ID to set multiview settings.
        :type project_id: int
        :Usage example:

         .. code-block:: python

            import os
            from dotenv import load_dotenv

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervise.ly'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'

            # Load secrets and create API object from .env file (recommended)
            # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
            load_dotenv(os.path.expanduser("~/supervisely.env"))
            api = sly.Api.from_env()

            api.project.set_multiview_settings(project_id=123)
        """

        self._set_custom_grouping_settings(
            id=project_id,
            group_images=True,
            tag_name=_MULTIVIEW_TAG_NAME,
            sync=False,
        )

    def remove_permanently(self, ids: Union[int, List]) -> dict:
        """
        Delete permanently projects with given IDs from the Supervisely server.

        !!! WARNING !!!
        Be careful, this method deletes data from the database, recovery is not possible.

        :param ids: IDs of projects in Supervisely.
        :type ids: Union[int, List]
        :return: Response content in JSON format
        :rtype: dict
        """

        if isinstance(ids, int):
            projects = [{"id": ids}]
        else:
            projects = [{"id": id} for id in ids]

        request_body = {
            ApiField.PROJECTS: projects,
            ApiField.PRESERVE_PROJECT_CARD: False,
        }

        response = self._api.post("projects.remove.permanently", request_body)

        return response.json()

    def get_list_all(
        self,
        filters: Optional[List[Dict[str, str]]] = None,
        skip_exported: Optional[bool] = True,
        sort: Optional[str] = None,
        sort_order: Optional[str] = None,
        per_page: Optional[int] = None,
        page: Union[int, Literal["all"]] = "all",
        account_type: Optional[str] = None,
    ) -> dict:
        """
        List all available projects from all available teams for the user that match the specified filtering criteria.

        :param filters: List of parameters for filtering the available Projects.
                        Every Dict must consist of keys:
                        - 'field': Takes values 'id', 'projectId', 'workspaceId', 'groupId', 'createdAt', 'updatedAt', 'type'
                        - 'operator': Takes values '=', 'eq', '!=', 'not', 'in', '!in', '>', 'gt', '>=', 'gte', '<', 'lt', '<=', 'lte'
                        - 'value': Takes on values according to the meaning of 'field' or null
        :type filters: List[Dict[str, str]], optional

        :param skip_exported: Determines whether to skip archived projects.
        :type skip_exported: bool, optional.

        :param sort: Specifies by which parameter to sort the project list.
                        Takes values 'id', 'name', 'size', 'createdAt', 'updatedAt'
        :type sort: str, optional

        :param sort_order: Determines which value to list from.
        :type sort_order: str, optional

        :param per_page: Number of first items found to be returned.
                        'None' will return the first page with a default size of 20000 projects.
        :type per_page: int, optional

        :param page: Page number, used to retrieve the following items if the number of them found is more than per_page.
                     The default value is 'all', which retrieves all available projects.
                     'None' will return the first page with projects, the amount of which is set in param 'per_page'.
        :type page: Union[int, Literal["all"]], optional

        :param account_type: Type of user account
        :type account_type: str, optional

        :return: Search response information and 'ProjectInfo' of all projects that are searched by a given criterion.
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
            projects = api.projects.get_list_all(filters, True)
            print(projects)
            # Output:
            # {
            #     "total": 2,
            #     "perPage": 20000,
            #     "pagesCount": 1,
            #     "entities": [ ProjectInfo(id = 22,
            #                       name = 'lemons_annotated',
            #                       description = None,
            #                       size = '861069',
            #                       readme = None,
            #                       workspace_id = 2,
            #                       images_count = None,
            #                       items_count = None,
            #                       datasets_count = None,
            #                       created_at = '2020-04-03T13:43:24.000Z',
            #                       updated_at = '2020-04-03T14:53:00.952Z',
            #                       type = 'images',
            #                       reference_image_url = None,
            #                       custom_data = None,
            #                       backup_archive = None,
            #                       teamd_id = 1),
            #                   ProjectInfo(id = 23,
            #                       name = 'lemons_test',
            #                       description = None,
            #                       size = '1177212',
            #                       readme = None,
            #                       workspace_id = 2,
            #                       images_count = None,
            #                       items_count = None,
            #                       datasets_count = None,
            #                       created_at = '2020-04-03T13:43:24.000Z',
            #                       updated_at = '2020-04-03T14:53:00.952Z',
            #                       type = 'images',
            #                       reference_image_url = None,
            #                       custom_data = None,
            #                       backup_archive = None),
            #                       teamd_id = 1)
            #                 ]
            # }

        """

        method = "projects.list.all"

        request_body = {}
        if filters is not None:
            request_body[ApiField.FILTER] = filters
        if skip_exported is not None:
            request_body[ApiField.SKIP_EXPORTED] = skip_exported
        if sort is not None:
            request_body[ApiField.SORT] = sort
        if sort_order is not None:
            request_body[ApiField.SORT_ORDER] = sort_order
        if per_page is not None:
            request_body[ApiField.PER_PAGE] = per_page
        if page is not None and page != "all":
            request_body[ApiField.PAGE] = page
        if account_type is not None:
            request_body[ApiField.ACCOUNT_TYPE] = account_type

        first_response = self._api.post(method, request_body).json()

        total = first_response.get(ApiField.TOTAL)
        per_page = first_response.get("perPage")
        pages_count = first_response.get("pagesCount")

        def _convert_entities(response_dict: dict):
            """
            Convert entities dict to ProjectInfo
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
