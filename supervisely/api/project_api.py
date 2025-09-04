# coding: utf-8
"""create/download/update :class:`Project<supervisely.project.project.Project>`"""

# docs
from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

from datetime import datetime, timedelta, timezone

from supervisely import logger
from supervisely._utils import (
    abs_url,
    compare_dicts,
    compress_image_url,
    get_unix_timestamp,
    is_development,
)
from supervisely.annotation.annotation import TagCollection
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.module_api import (
    ApiField,
    CloneableModuleApi,
    RemoveableModuleApi,
    UpdateableModule,
)
from supervisely.io.env import upload_count, uploaded_ids
from supervisely.io.json import dump_json_file, load_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_meta import ProjectMetaJsonFields as MetaJsonF
from supervisely.project.project_settings import (
    ProjectSettings,
    ProjectSettingsJsonFields,
)
from supervisely.project.project_type import (
    _LABEL_GROUP_TAG_NAME,
    _METADATA_SYSTEM_KEY,
    _METADATA_TIMESTAMP_KEY,
    _METADATA_VALIDATION_SCHEMA_KEY,
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
    import_settings: dict
    version: dict
    created_by_id: int
    embeddings_enabled: Optional[bool] = None
    embeddings_updated_at: Optional[str] = None
    embeddings_in_progress: Optional[bool] = None
    local_entities_count: Optional[int] = None
    remote_entities_count: Optional[int] = None

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
        # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

        project_id = 1951
        project_info = api.project.get_info_by_id(project_id)
    """

    debug_messages_sent = {"get_list_versions": False}

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
                        reference_image_url='http://app.supervisely.com/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg',
                        custom_data={},
                        backup_archive={},
                        team_id=2,
                        import_settings={}
                        version={'id': 260, 'version': 3}
                        created_by_id=7,
                        embeddings_enabled=False,
                        embeddings_updated_at=None,
                        embeddings_in_progress=False,
                        local_entities_count=10,
                        remote_entities_count=0
                        )
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
            ApiField.IMPORT_SETTINGS,
            ApiField.VERSION,
            ApiField.CREATED_BY_ID,
            ApiField.EMBEDDINGS_ENABLED,
            ApiField.EMBEDDINGS_UPDATED_AT,
            ApiField.EMBEDDINGS_IN_PROGRESS,
            ApiField.LOCAL_ENTITIES_COUNT,
            ApiField.REMOTE_ENTITIES_COUNT,
        ]

    @staticmethod
    def info_sequence_for_listing():
        """
        NamedTuple ProjectInfo fields available for listing operations.

        This subset includes only fields that are available in the `projects.list` API endpoint.
        For complete project information, use `get_info_by_id()`.

        :return: List of API field names available for listing
        :rtype: List[str]
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.DESCRIPTION,
            ApiField.SIZE,
            ApiField.README,
            ApiField.WORKSPACE_ID,
            ApiField.IMAGES_COUNT,  # for compatibility with existing code
            ApiField.DATASETS_COUNT,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.TYPE,
            ApiField.REFERENCE_IMAGE_URL,
            ApiField.CUSTOM_DATA,
            ApiField.BACKUP_ARCHIVE,
            ApiField.TEAM_ID,
            ApiField.IMPORT_SETTINGS,
            ApiField.EMBEDDINGS_ENABLED,
            ApiField.EMBEDDINGS_UPDATED_AT,
            ApiField.EMBEDDINGS_IN_PROGRESS,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **ProjectInfo**.
        """
        return "ProjectInfo"

    def __init__(self, api):
        from supervisely.project.data_version import DataVersion

        CloneableModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)
        self.version = DataVersion(api)

    def get_list(
        self,
        workspace_id: Optional[int] = None,
        filters: Optional[List[Dict[str, str]]] = None,
        fields: List[str] = [],
        team_id: Optional[int] = None,
    ) -> List[ProjectInfo]:
        """
        List of Projects in the given Workspace.

        *NOTE*: Version information is not available while getting list of projects.
        If you need version information, use :func:`get_info_by_id`.

        :param workspace_id: Workspace ID in which the Projects are located.
        :type workspace_id: int, optional
        :param filters: List of params to sort output Projects.
        :type filters: List[dict], optional
        :param fields: The list of api fields which will be returned with the response. You must specify all fields you want to receive, not just additional ones.
        :type fields: List[str]
        :param team_id: Team ID in which the Projects are located.
        :type team_id: int, optional
        :return: List of all projects with information for the given Workspace. See :class:`info_sequence<info_sequence>`
        :rtype: :class: `List[ProjectInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            workspace_id = 58

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
            #             backup_archive={},
            #             import_settings={}
            #           ),
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
            #             reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg',
            #             custom_data={},
            #             backup_archive={},
            #             import_settings={}
            #           )
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
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg',
            #                     custom_data={},
            #                     backup_archive={},
            #                     import_settings={}
            #                   )
            # ]

        """
        if team_id is not None and workspace_id is not None:
            raise ValueError(
                "team_id and workspace_id cannot be used together. Please provide only one of them."
            )

        method = "projects.list"

        debug_message = "While getting list of projects, the following fields are not available: "

        if ApiField.VERSION in fields:
            fields.remove(ApiField.VERSION)
            if self.debug_messages_sent.get("get_list_versions", False) is False:
                self.debug_messages_sent["get_list_versions"] = True
                logger.debug(debug_message + "version. ")

        default_fields = [
            ApiField.ID,
            ApiField.WORKSPACE_ID,
            ApiField.TITLE,
            ApiField.DESCRIPTION,
            ApiField.SIZE,
            ApiField.README,
            ApiField.TYPE,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.CUSTOM_DATA,
            ApiField.GROUP_ID,
            ApiField.CREATED_BY_ID[0][0],
        ]

        if fields:
            merged_fields = list(set(default_fields + fields))
            fields = list(dict.fromkeys(merged_fields))

        data = {
            ApiField.FILTER: filters or [],
            ApiField.FIELDS: fields,
        }
        if workspace_id is not None:
            data[ApiField.WORKSPACE_ID] = workspace_id
        if team_id is not None:
            data[ApiField.GROUP_ID] = team_id

        return self.get_list_all_pages(method, data)

    def get_info_by_id(
        self,
        id: int,
        expected_type: Optional[str] = None,
        raise_error: bool = False,
        extra_fields: Optional[List[str]] = None,
    ) -> ProjectInfo:
        """
        Get Project information by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :param expected_type: Expected ProjectType.
        :type expected_type: ProjectType, optional
        :param raise_error: If True raise error if given name is missing in the Project, otherwise skips missing names.
        :type raise_error: bool, optional
        :param extra_fields: List of extra fields to include in the response.
        :type extra_fields: list[str], optional
        :raises: Error if type of project is not None and != expected type
        :return: Information about Project. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`ProjectInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            project_id = 1951

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg',
            #                     custom_data={},
            #                     backup_archive={},
            #                     import_settings={}
            #                   )


        """
        fields = None
        if extra_fields is not None:
            fields = {ApiField.EXTRA_FIELDS: extra_fields}
        info = self._get_info_by_id(id, "projects.info", fields=fields)
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

        Version information is not available while getting project by name.
        If you need version information, use :func:`get_info_by_id`.

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
            #                     reference_image_url='http://78.46.75.100:38585/h5un6l2bnaz1vj8a9qgms4-public/images/original/...jpg',
            #                     custom_data={},
            #                     backup_archive={},
            #                     import_settings={}
            #                   )
        """
        try:
            fields = self.info_sequence_for_listing()
            info = super().get_info_by_name(parent_id, name, fields)
        except Exception as e:
            logger.trace(
                f"Failed to get info by name with all available fields for 'projects.list' endpoint: {e} "
                "Falling back to minimal fields (id) and get_info_by_id()."
            )
            fields = [ApiField.ID]
            info = super().get_info_by_name(parent_id, name, fields)
            if info is None:
                if raise_error:
                    raise ProjectNotFound(
                        f"Project with name {name!r} not found in workspace {parent_id!r}."
                    )
                else:
                    return None
            info = self.get_info_by_id(info.id)
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
            mtag_name = None

            if json_settings.get("groupImagesByTagId") is not None:
                for tag in json_response["tags"]:
                    if tag["id"] == json_settings["groupImagesByTagId"]:
                        mtag_name = tag["name"]
                        break

            json_response[MetaJsonF.PROJECT_SETTINGS] = ProjectSettings(
                multiview_enabled=json_settings.get("groupImages", False),
                multiview_tag_name=mtag_name,
                multiview_tag_id=json_settings.get("groupImagesByTagId"),
                multiview_is_synced=json_settings.get("groupImagesSync", False),
                labeling_interface=json_settings.get(ProjectSettingsJsonFields.LABELING_INTERFACE),
            ).to_json()

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
            #                     reference_image_url=None,
            #                     custom_data={},
            #                     backup_archive={},
            #                     import_settings={}
            #                   )

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

    def update_meta(self, id: int, meta: Union[ProjectMeta, Dict]) -> ProjectMeta:
        """
        Updates given Project with given ProjectMeta.

        :param id: Project ID in Supervisely.
        :type id: int
        :param meta: ProjectMeta object or ProjectMeta in JSON format.
        :type meta: :class:`ProjectMeta` or dict

        :return: ProjectMeta
        :rtype: :class: `ProjectMeta`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            lemons_proj_id = 1951
            kiwis_proj_id = 1952

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

        m.project_settings.validate(m)
        response = self._api.post(
            "projects.meta.update", {ApiField.ID: id, ApiField.META: m.to_json()}
        )
        try:
            tmp = ProjectMeta.from_json(data=response.json())
            m = tmp.clone(project_type=m.project_type, project_settings=m.project_settings)
        except KeyError:
            pass  # handle old instances <6.8.69: response.json()=={'success': True}

        if m.project_settings is not None:
            s = m.project_settings
            mtag_name = s.multiview_tag_name
            mtag_id = s.multiview_tag_id
            if mtag_name is None:
                mtag_name = m.get_tag_name_by_id(s.multiview_tag_id)

            if mtag_name is not None:  # (tag_id, tag_name)==(None, None) is OK but no group
                new_m = ProjectMeta.from_json(self.get_meta(id))
                group_tag = new_m.get_tag_meta(mtag_name)
                mtag_id = None if group_tag is None else group_tag.sly_id

            new_s = s.clone(
                multiview_tag_name=mtag_name,
                multiview_tag_id=mtag_id,
            )
            labeling_interface = new_s.labeling_interface
            m = m.clone(project_settings=new_s)
            settings_json = {
                "groupImages": new_s.multiview_enabled,
                "groupImagesByTagId": new_s.multiview_tag_id,
                "groupImagesSync": new_s.multiview_is_synced,
            }
            if labeling_interface is not None:
                settings_json[ProjectSettingsJsonFields.LABELING_INTERFACE] = labeling_interface
            self.update_settings(id, settings_json)

        return m

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_imgs_count = api.project.get_images_count(project_id)
            print(project_imgs_count)
            # Output: 24
        """
        datasets = self._api.dataset.get_list(id, recursive=True)
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            project_url = api.project.url(project_id)
            print(project_url)
            # Output: http://supervisely.com/projects/1951/datasets
        """
        res = f"projects/{id}/datasets"
        if is_development():
            res = abs_url(res)
        return res

    def update_custom_data(self, id: int, data: Dict, silent: bool = False) -> Dict:
        """
        Updates custom data of the Project by ID.

        IMPORTANT: This method replaces the current custom data with the provided one.
        If you want to extend the custom data or update specific key-value pairs,
        use :func:get_custom_data first to retrieve the existing data,
        then modify it accordingly before calling this method.

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

    def get_custom_data(self, id: int) -> Dict[Any, Any]:
        """Returns custom data of the Project by ID.
        Custom data is a dictionary that can be used to store any additional information.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: Custom data of the Project
        :rtype: :class:`dict`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123456

            custom_data = api.project.get_custom_data(project_id)

            print(custom_data) # Output: {'key': 'value'}
        """
        return self.get_info_by_id(id).custom_data

    def _get_system_custom_data(self, id: int) -> Dict[Any, Any]:
        """Returns system custom data of the Project by ID.
        System custom data is just a part of custom data that is used to store system information
        and obtained by the key `_METADATA_SYSTEM_KEY`.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: System custom data of the Project
        :rtype: :class:`dict`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123456

            system_custom_data = api.project._get_system_custom_data(project_id)

            print(system_custom_data)
        """
        return self.get_info_by_id(id).custom_data.get(_METADATA_SYSTEM_KEY, {})

    def get_validation_schema(self, id: int, use_caching: bool = False) -> Optional[Dict[Any, Any]]:
        """Returns validation schema of the Project by ID.
        Validation schema is a dictionary that can be used to validate metadata of each entity in the project
        if corresnpoding schema is provided.
        If using caching, the schema will be loaded from the cache if available.
        Use cached version only in scenarios when the schema is not expected to change,
        otherwise it may lead to checks with outdated schema.

        :param id: Project ID in Supervisely.
        :type id: int
        :param use_caching: If True, uses cached version of the schema if available.
            NOTE: This may lead to checks with outdated schema. Use with caution.
            And only in scenarios when the schema is not expected to change.
        :return: Validation schema of the Project
        :rtype: :class:`dict`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123456

            validation_schema = api.project.get_validation_schema(project_id)

            print(validation_schema) # Output: {'key': 'Description of the field'}
        """
        SCHEMA_DIFF_THRESHOLD = 60 * 60  # 1 hour
        json_cache_filename = os.path.join(os.getcwd(), f"{id}_validation_schema.json")

        if use_caching:
            if os.path.isfile(json_cache_filename):
                try:
                    schema = load_json_file(json_cache_filename)
                    timestamp = schema.pop(_METADATA_TIMESTAMP_KEY, 0)

                    if get_unix_timestamp() - timestamp < SCHEMA_DIFF_THRESHOLD:
                        return schema
                except RuntimeError:
                    pass

        schema = self._get_system_custom_data(id).get(_METADATA_VALIDATION_SCHEMA_KEY)
        if schema and use_caching:
            schema_with_timestamp = deepcopy(schema)
            schema_with_timestamp[_METADATA_TIMESTAMP_KEY] = get_unix_timestamp()
            dump_json_file(schema_with_timestamp, json_cache_filename)

        return schema

    def _edit_validation_schema(
        self, id: int, schema: Optional[Dict[Any, Any]] = None
    ) -> Dict[Any, Any]:
        """Edits validation schema of the Project by ID.
        Do not use this method directly, use `set_validation_schema` or `remove_validation_schema` instead.

        :param id: Project ID in Supervisely.
        :type id: int
        :param schema: Validation schema to set. If None, removes validation schema.
        :type schema: dict, optional
        :return: Project information in dict format
        :rtype: :class:`dict`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123456

            schema = {'key': 'Description of the field'}

            api.project._edit_validation_schema(project_id, schema) #Set new validation schema.
            api.project._edit_validation_schema(project_id) #Remove validation schema.
        """
        custom_data = self.get_custom_data(id)
        system_data = custom_data.setdefault(_METADATA_SYSTEM_KEY, {})

        if not schema:
            system_data.pop(_METADATA_VALIDATION_SCHEMA_KEY, None)
        else:
            system_data[_METADATA_VALIDATION_SCHEMA_KEY] = schema
        return self.update_custom_data(id, custom_data)

    def set_validation_schema(self, id: int, schema: Dict[Any, Any]) -> Dict[Any, Any]:
        """Sets validation schema of the Project by ID.
        NOTE: This method will overwrite existing validation schema. To extend existing schema,
        use `get_validation_schema` first to get current schema, then update it and use this method to set new schema.

        :param id: Project ID in Supervisely.
        :type id: int
        :param schema: Validation schema to set.
        :type schema: dict
        :return: Project information in dict format
        :rtype: :class:`dict`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123456

            schema = {'key': 'Description of the field'}

            api.project.set_validation_schema(project_id, schema)
        """
        return self._edit_validation_schema(id, schema)

    def remove_validation_schema(self, id: int) -> Dict[Any, Any]:
        """Removes validation schema of the Project by ID.

        :param id: Project ID in Supervisely.
        :type id: int
        :return: Project information in dict format
        :rtype: :class:`dict`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123456

            api.project.remove_validation_schema(project_id)
        """
        return self._edit_validation_schema(id)

    def validate_entities_schema(
        self, id: int, strict: bool = False
    ) -> List[Dict[str, Union[id, str, List[str], List[Any]]]]:
        """Validates entities of the Project by ID using validation schema.
        Returns list of entities that do not match the schema.

        Example of the returned list:

        [
            {
                "entity_id": 123456,
                "entity_name": "image.jpg",
                "missing_fields": ["location"],
                "extra_fields": ["city.name"] <- Nested field (field "name" of the field "city")
            }
        ]

        :param id: Project ID in Supervisely.
        :type id: int
        :param strict: If strict is disabled, only checks if the entity has all the fields from the schema.
            Any extra fields in the entity will be ignored and will not be considered as an error.
            If strict is enabled, checks that the entity custom data is an exact match to the schema.
        :type strict: bool, optional
        :return: List of dictionaries with information about entities that do not match the schema.
        :rtype: :class:`List[Dict[str, Union[id, str, List[str], List[Any]]]`

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123456

            incorrect_entities = api.project.validate_entities_schema(project_id)

            for entity in incorrect_entities:
                print(entity["entity_id"], entity["entity_name"]) # Output: 123456, 'image.jpg'
        """
        validation_schema = self.get_validation_schema(id)
        if not validation_schema:
            raise ValueError("Validation schema is not set for this project.")

        info = self.get_info_by_id(id)
        listing_method = {
            # Mapping of project type to listing method.
            ProjectType.IMAGES.value: self._api.image.get_list,
        }
        custom_data_properties = {
            # Mapping of project type to custom data property name.
            # Can be different for different project types.
            ProjectType.IMAGES.value: "meta"
        }

        listing_method = listing_method.get(info.type)
        custom_data_property = custom_data_properties.get(info.type)

        if not listing_method or not custom_data_property:
            # TODO: Add support for other project types.
            raise NotImplementedError("Validation schema is not supported for this project type.")

        entities = listing_method(project_id=id)
        incorrect_entities = []

        for entity in entities:
            custom_data = getattr(entity, custom_data_property)
            missing_fields, extra_fields = compare_dicts(
                validation_schema, custom_data, strict=strict
            )
            if missing_fields or extra_fields:
                entry = {
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "missing_fields": missing_fields,
                    "extra_fields": extra_fields,
                }
                incorrect_entities.append(entry)

        return incorrect_entities

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

        :param id: Project ID, where images grouping will be enabled.
        :type id: int
        :param enable: if True groups images by given tag name, otherwise disables images grouping.
        :type enable: bool
        :param tag_name: Name of the tag. Images will be grouped by this tag.
        :type tag_name: str
        :param sync: Enable the syncronization views mode in the grouping settings. By default, `False`
        :type sync: bool
        """
        project_meta_json = self.get_meta(id)
        project_meta = ProjectMeta.from_json(project_meta_json)
        group_tag_meta = project_meta.get_tag_meta(tag_name)
        if group_tag_meta is None:
            raise RuntimeError(f"The group tag '{tag_name}' doesn't exist in the given project.")
        elif group_tag_meta.value_type != TagValueType.ANY_STRING:
            raise RuntimeError(
                f"The tag value type should be '{TagValueType.ANY_STRING}' for images grouping. The provided type: '{group_tag_meta.value_type}'"
            )
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
            #             type='images',
            #             ...
            #             ),
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
            #                     type='images',
            #                     ...
            #                     )
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
            filer_to = {
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
            # api = sly.Api(server_address="https://app.supervisely.com", token="4r47N...xaTatb")

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

    def _set_custom_grouping_settings(
        self,
        id: int,
        group_images: bool,
        tag_name: str,
        sync: bool,
        label_group_tag_name: str = None,
    ) -> None:
        """Sets the project settings for custom grouping.

        :param id: Project ID to set custom grouping settings.
        :type id: int
        :param group_images: if True enables images grouping by tag
        :type group_images: bool
        :param tag_name: Name of the tag. Images will be grouped by this tag
        :type tag_name: str
        :param sync: if True images will have synchronized view and labeling
        :type sync: bool
        :param label_group_tag_name: Name of the tag. Labels will be grouped by this tag
        :type label_group_tag_name: str
        :raises ValueError: if tag value type is not 'any_string'
        :return: None
        :rtype: :class:`NoneType`
        """
        meta = ProjectMeta.from_json(self.get_meta(id, with_settings=True))
        existing_tag_meta = meta.get_tag_meta(tag_name)
        need_update = False
        if existing_tag_meta is not None:
            if existing_tag_meta.value_type != TagValueType.ANY_STRING:
                raise ValueError(
                    f"Tag '{tag_name}' should have value type 'any_string', "
                    f"but got '{existing_tag_meta.value_type}' value type."
                )
        else:
            new_tag_meta = TagMeta(tag_name, TagValueType.ANY_STRING)
            meta = meta.add_tag_meta(new_tag_meta)
            need_update = True
        if label_group_tag_name is not None:
            label_group_tag_meta = meta.get_tag_meta(label_group_tag_name)
            if label_group_tag_meta is not None:
                if label_group_tag_meta.value_type != TagValueType.ANY_STRING:
                    raise ValueError(
                        f"Tag '{label_group_tag_name}' should have value type 'any_string', "
                        f"but got '{label_group_tag_meta.value_type}' value type."
                    )
            else:
                label_group_tag_meta = TagMeta(label_group_tag_name, TagValueType.ANY_STRING)
                meta = meta.add_tag_meta(label_group_tag_meta)
                need_update = True
        if need_update:
            self.update_meta(id, meta)

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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
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
            label_group_tag_name=_LABEL_GROUP_TAG_NAME,
        )

    def remove_permanently(
        self, ids: Union[int, List], batch_size: int = 50, progress_cb=None
    ) -> List[dict]:
        """
        !!! WARNING !!!
        Be careful, this method deletes data from the database, recovery is not possible.

        Delete permanently projects with given IDs from the Supervisely server.
        All project IDs must belong to the same team.
        Therefore, it is necessary to sort IDs before calling this method.

        :param ids: IDs of projects in Supervisely.
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
            projects = [{ApiField.ID: ids}]
        else:
            projects = [{ApiField.ID: id} for id in ids]

        batches = [projects[i : i + batch_size] for i in range(0, len(projects), batch_size)]
        responses = []
        for batch in batches:
            request_body = {
                ApiField.PROJECTS: batch,
                ApiField.PRESERVE_PROJECT_CARD: False,
            }
            response = self._api.post("projects.remove.permanently", request_body)
            if progress_cb is not None:
                progress_cb(len(batch))
            responses.append(response.json())
        return responses

    def get_list_all(
        self,
        filters: Optional[List[Dict[str, str]]] = None,
        skip_exported: Optional[bool] = True,
        sort: Optional[str] = None,
        sort_order: Optional[str] = None,
        per_page: Optional[int] = None,
        page: Union[int, Literal["all"]] = "all",
        account_type: Optional[str] = None,
        extra_fields: Optional[List[str]] = None,
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

        :param account_type: (Deprecated) Type of user account
        :type account_type: str, optional

        :param extra_fields: List of additional fields to be included in the response.
        :type extra_fields: List[str], optional

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
            #                       team_id = 1,
            #                       import_settings = {},
            #                   ),
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
            #                       team_id = 1,
            #                       import_settings = {},
            #                   )
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
            logger.warning(
                "The 'account_type' parameter is deprecated. The result will not be filtered by account type. To filter received ProjectInfos, you could use the 'team_id' from the ProjectInfo object to get TeamInfo and check the account type."
            )
        if extra_fields is not None:
            request_body[ApiField.EXTRA_FIELDS] = extra_fields

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

    def enable_embeddings(self, id: int, silent: bool = True) -> None:
        """
        Enable embeddings for the project.

        :param id: Project ID
        :type id: int
        :param silent: Determines whether the `updatedAt` timestamp of the Project should be updated or not, if False - update `updatedAt`
        :type silent: bool
        :return: None
        :rtype: :class:`NoneType`
        """
        self._api.post(
            "projects.editInfo",
            {ApiField.ID: id, ApiField.EMBEDDINGS_ENABLED: True, ApiField.SILENT: silent},
        )

    def disable_embeddings(self, id: int, silent: bool = True) -> None:
        """
        Disable embeddings for the project.

        :param id: Project ID
        :type id: int
        :param silent: Determines whether the `updatedAt` timestamp of the Poject should be updated or not, if False - update `updatedAt`
        :type silent: bool
        :return: None
        :rtype: :class:`NoneType`
        """
        self._api.post(
            "projects.editInfo",
            {ApiField.ID: id, ApiField.EMBEDDINGS_ENABLED: False, ApiField.SILENT: silent},
        )

    def is_embeddings_enabled(self, id: int) -> bool:
        """
        Check if embeddings are enabled for the project.

        :param id: Project ID
        :type id: int
        :return: True if embeddings are enabled, False otherwise.
        :rtype: bool
        """
        info = self.get_info_by_id(id, extra_fields=[ApiField.EMBEDDINGS_ENABLED])
        return info.embeddings_enabled

    def set_embeddings_in_progress(
        self, id: int, in_progress: bool, error_message: Optional[str] = None
    ) -> None:
        """
        Set embeddings in progress status for the project.
        This method is used to indicate whether embeddings are currently being created for the project.

        :param id: Project ID
        :type id: int
        :param in_progress: Status to set. If True, embeddings are in progress right now.
        :type in_progress: bool
        :param error_message: Optional error message to provide additional context.
        :type error_message: Optional[str]
        :return: None
        :rtype: :class:`NoneType`
        """
        data = {ApiField.ID: id, ApiField.EMBEDDINGS_IN_PROGRESS: in_progress}
        if error_message is not None:
            data[ApiField.ERROR_MESSAGE] = error_message
        self._api.post("projects.embeddings-in-progress.update", data)

    def get_embeddings_in_progress(self, id: int) -> bool:
        """
        Get the embeddings in progress status for the project.
        This method checks whether embeddings are currently being created for the project.

        :param id: Project ID
        :type id: int
        :return: True if embeddings are in progress, False otherwise.
        :rtype: bool
        """
        info = self.get_info_by_id(id, extra_fields=[ApiField.EMBEDDINGS_IN_PROGRESS])
        if info is None:
            raise RuntimeError(f"Project with ID {id} not found.")
        if not hasattr(info, "embeddings_in_progress"):
            raise RuntimeError(
                f"Project with ID {id} does not have 'embeddings_in_progress' field in its info."
            )
        return info.embeddings_in_progress

    def set_embeddings_updated_at(
        self, id: int, timestamp: Optional[str] = None, silent: bool = True
    ) -> None:
        """
        Set the timestamp when embeddings were last updated for the project.
        If no timestamp is provided, uses the current UTC time.

        :param id: Project ID
        :type id: int
        :param timestamp: ISO format timestamp (YYYY-MM-DDTHH:MM:SS.fffffZ). If None, current UTC time is used.
        :type timestamp: Optional[str]
        :param silent: Determines whether the `updatedAt` timestamp of the Project should be updated or not, if False - update `updatedAt`
        :type silent: bool
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python


            api = sly.Api.from_env()
            project_id = 123

            # Set current time as embeddings update timestamp
            api.project.set_embeddings_updated_at(project_id)

            # Set specific timestamp
            api.project.set_embeddings_updated_at(project_id, "2025-06-01T10:30:45.123456Z")
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        self._api.post(
            "projects.editInfo",
            {ApiField.ID: id, ApiField.EMBEDDINGS_UPDATED_AT: timestamp, ApiField.SILENT: silent},
        )

    def get_embeddings_updated_at(self, id: int) -> Optional[str]:
        """
        Get the timestamp when embeddings were last updated for the project.

        :param id: Project ID
        :type id: int
        :return: ISO format timestamp (YYYY-MM-DDTHH:MM:SS.fffZ) or None if not set.
        :rtype: Optional[str]
        :Usage example:

         .. code-block:: python

            api = sly.Api.from_env()
            project_id = 123

            # Get embeddings updated timestamp
            updated_at = api.project.get_embeddings_updated_at(project_id)
            print(updated_at)  # Output: "2025-06-01T10:30:45.123Z" or None
        """
        info = self.get_info_by_id(id, extra_fields=[ApiField.EMBEDDINGS_UPDATED_AT])
        if info is None:
            raise RuntimeError(f"Project with ID {id} not found.")
        if not hasattr(info, "embeddings_updated_at"):
            raise RuntimeError(
                f"Project with ID {id} does not have 'embeddings_updated_at' field in its info."
            )
        return info.embeddings_updated_at

    def perform_ai_search(
        self,
        project_id: int,
        dataset_id: Optional[int] = None,
        image_id: Optional[Union[int, List[int]]] = None,
        prompt: Optional[str] = None,
        method: Optional[Literal["centroids", "random"]] = None,
        limit: int = 100,
        clustering_method: Optional[Literal["kmeans", "dbscan"]] = None,
        num_clusters: Optional[int] = None,
        image_id_scope: Optional[List[int]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[int]:
        """
        Send AI search request to initiate search process.
        This method allows you to search for similar images in a project using either a text prompt, an image ID, or a method type.
        It is mutually exclusive, meaning you can only provide one of the parameters: `prompt`, `image_id`, or `method`.

        :param project_id: ID of the Project
        :type project_id: int
        :param dataset_id: ID of the Dataset. If not None - search will be limited to this dataset.
        :type dataset_id: Optional[int]
        :param image_id: ID(s) of the Image(s). Searches for images similar to the specified image(s).
        :type image_id: Optional[Union[int, List[int]]]
        :param prompt: Text prompt for search request. Searches for similar images based on a text description.
        :type prompt: Optional[str]
        :param method: Activates diverse search using one of the following methods: "centroids", "random".
        :type method: Optional[Literal["centroids", "random"]]
        :param limit: Limit for search request
        :type limit: int
        :param clustering_method: Method for clustering results. Can be "kmeans" or "dbscan". If None, no clustering is applied.
        :type clustering_method: Optional[Literal["kmeans", "dbscan"]]
        :param num_clusters: Number of clusters to create if clustering_method is specified. Required for "kmeans" method.
        :type num_clusters: Optional[int]
        :param image_id_scope: List of image IDs to limit the search scope. If None, the search will be performed across all images in the project if other filters are not set.
        :type image_id_scope: Optional[List[int]]
        :param threshold: Threshold for similarity. If provided, only images with similarity above this threshold will be returned.
        :type threshold: Optional[float]
        :return: Entitites Collection ID of the search results, or None if no collection was created.
        :rtype: Optional[int]
        :raises ValueError: only one of `prompt`, `image_id` or `method`must be provided, and `method` must be one of the allowed values.
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            project_id = 123
            image_id = 789
            prompt = "person with a dog"

            # Search with text prompt
            collection_id = api.project.perform_ai_search(
                project_id=project_id,
                prompt=prompt,
            )

            # Search with method
            collection_id = api.project.perform_ai_search(
                project_id=project_id,
                method="centroids",
            )

            # Search with image ID
            collection_id = api.project.perform_ai_search(
                project_id=project_id,
                image_id=image_id,
            )
        """

        # Check that only one of prompt, method, or image_id is provided
        provided_params = sum([prompt is not None, method is not None, image_id is not None])
        if provided_params != 1:
            raise ValueError(
                "Must provide exactly one of 'prompt', 'method', or 'image_id' parameters. They are mutually exclusive."
            )

        if prompt is None and method is None and image_id is None:
            raise ValueError("Must provide either 'prompt', 'method', or 'image_id' parameter.")

        # Validate method values
        if method is not None and method not in ["centroids", "random"]:
            raise ValueError("Method must be either 'centroids' or 'random'.")

        request_body = {
            ApiField.PROJECT_ID: project_id,
            ApiField.LIMIT: limit,
            ApiField.UNIQUE_ITEMS: limit,  # the same as limit, but for diverse search
        }

        if dataset_id is not None:
            request_body[ApiField.DATASET_ID] = dataset_id

        if prompt is not None:
            request_body[ApiField.PROMPT] = prompt

        if image_id is not None:
            if prompt is not None or method is not None:
                raise ValueError("If 'image_id' is provided, 'prompt' and 'method' must be None.")
            if isinstance(image_id, int):
                image_id = [image_id]
            if not isinstance(image_id, list):
                raise ValueError("image_id must be a list of image IDs.")
            request_body[ApiField.IMAGE_IDS] = image_id

        if method is not None:
            if image_id is not None or prompt is not None:
                raise ValueError("If 'method' is provided, 'image_id' and 'prompt' must be None.")
            request_body[ApiField.METHOD] = method

        if clustering_method is not None:
            if clustering_method not in ["kmeans", "dbscan"]:
                raise ValueError("Clustering method must be either 'kmeans' or 'dbscan'.")
            request_body[ApiField.CLUSTERING_METHOD] = clustering_method

        if num_clusters is not None:
            if clustering_method != "kmeans":
                raise ValueError(
                    "Number of clusters is only applicable for 'kmeans' clustering method."
                )
            request_body[ApiField.NUMBER_OF_CLUSTERS] = num_clusters

        if image_id_scope is not None:
            if not isinstance(image_id_scope, list):
                raise ValueError("image_id_scope must be a list of image IDs.")
            request_body[ApiField.RESTRICTED_IMAGE_IDS] = image_id_scope

        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValueError("Threshold must be a number.")
            request_body[ApiField.THRESHOLD] = threshold

        response = self._api.post("embeddings.send-ai-search", request_body)
        return response.json().get(ApiField.COLLECTION_ID, None)

    def calculate_embeddings(self, id: int) -> None:
        """
        Calculate embeddings for the project.
        This method is used to calculate embeddings for all images in the project.

        To check status of embeddings calculation, use :meth:`get_embeddings_in_progress`

        :param id: Project ID
        :type id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            project_id = 123

            # Calculate embeddings for the project
            api.project.calculate_embeddings(project_id)
        """
        self._api.post("embeddings.calculate-project-embeddings", {ApiField.PROJECT_ID: id})

    def recreate_structure_generator(
        self,
        src_project_id: int,
        dst_project_id: Optional[int] = None,
        dst_project_name: Optional[str] = None,
    ) -> Generator[Tuple[DatasetInfo, DatasetInfo], None, None]:
        """This method can be used to recreate a project with hierarchial datasets (without the data itself) and
        yields the tuple of source and destination DatasetInfo objects.

        :param src_project_id: Source project ID
        :type src_project_id: int
        :param dst_project_id: Destination project ID
        :type dst_project_id: int, optional
        :param dst_project_name: Name of the destination project. If `dst_project_id` is None, a new project will be created with this name. If `dst_project_id` is provided, this parameter will be ignored.
        :type dst_project_name: str, optional

        :return: Generator of tuples of source and destination DatasetInfo objects
        :rtype: Generator[Tuple[DatasetInfo, DatasetInfo], None, None]

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            src_project_id = 123
            dst_project_id = api.project.create("new_project", "images").id

            for src_ds, dst_ds in api.project.recreate_structure_generator(src_project_id, dst_project_id):
                print(f"Recreated dataset {src_ds.id} -> {dst_ds.id}")
                # Implement your logic here to process the datasets.
        """
        if dst_project_id is None:
            src_project_info = self._api.project.get_info_by_id(src_project_id)
            dst_project_info = self._api.project.create(
                src_project_info.workspace_id,
                dst_project_name or f"Recreation of {src_project_info.name}",
                src_project_info.type,
                src_project_info.description,
                change_name_if_conflict=True,
            )
            dst_project_id = dst_project_info.id

        datasets = self._api.dataset.get_list(src_project_id, recursive=True, include_custom_data=True)
        src_to_dst_ids = {}

        for src_dataset_info in datasets:
            dst_dataset_info = self._api.dataset.create(
                dst_project_id,
                src_dataset_info.name,
                description=src_dataset_info.description,
                parent_id=src_to_dst_ids.get(src_dataset_info.parent_id),
                custom_data=src_dataset_info.custom_data,
            )
            src_to_dst_ids[src_dataset_info.id] = dst_dataset_info.id

            yield src_dataset_info, dst_dataset_info

    def recreate_structure(
        self,
        src_project_id: int,
        dst_project_id: Optional[int] = None,
        dst_project_name: Optional[str] = None,
    ) -> Tuple[List[DatasetInfo], List[DatasetInfo]]:
        """This method can be used to recreate a project with hierarchial datasets (without the data itself).

        :param src_project_id: Source project ID
        :type src_project_id: int
        :param dst_project_id: Destination project ID
        :type dst_project_id: int, optional
        :param dst_project_name: Name of the destination project. If `dst_project_id` is None, a new project will be created with this name. If `dst_project_id` is provided, this parameter will be ignored.
        :type dst_project_name: str, optional

        :return: Destination project ID
        :rtype: int

        :Usage example:

        .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()

            src_project_id = 123
            dst_project_name = "New Project"

            dst_project_id = api.project.recreate_structure(src_project_id, dst_project_name=dst_project_name)
            print(f"Recreated project {src_project_id} -> {dst_project_id}")
        """
        infos = []
        for src_info, dst_info in self.recreate_structure_generator(
            src_project_id, dst_project_id, dst_project_name
        ):
            infos.append((src_info, dst_info))

        return infos

    def add_import_history(self, id: int, task_id: int) -> None:
        """
        Adds import history to project info. Gets task info and adds it to project custom data.

        :param id: Project ID
        :type id: int
        :param task_id: Task ID
        :type task_id: int
        :return: None
        :rtype: :class:`NoneType`
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
            task_id = 456
            api.project.add_import_history(project_id, task_id)
        """

        task_info = self._api.task.get_info_by_id(task_id)
        module_id = task_info.get("meta", {}).get("app", {}).get("moduleId")
        slug = None
        if module_id is not None:
            module_info = self._api.app.get_ecosystem_module_info(module_id)
            slug = module_info.slug

        items_count = upload_count()
        items_count = {int(k): v for k, v in items_count.items()}
        uploaded_images = uploaded_ids()
        uploaded_images = {int(k): v for k, v in uploaded_images.items()}
        total_items = sum(items_count.values()) if len(items_count) > 0 else 0
        app = task_info.get("meta", {}).get("app")
        app_name = app.get("name") if app else None
        app_version = app.get("version") if app else None
        data = {
            "task_id": task_id,
            "app": {"name": app_name, "version": app_version},
            "slug": slug,
            "status": task_info.get(ApiField.STATUS),
            "user_id": task_info.get(ApiField.USER_ID),
            "team_id": task_info.get(ApiField.TEAM_ID),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "source_state": task_info.get("settings", {}).get("message", {}).get("state"),
            "items_count": total_items,
            "datasets": [
                {
                    "id": ds,
                    "items_count": items_count[ds],
                    "uploaded_images": uploaded_images.get(ds, []),
                }
                for ds in items_count.keys()
            ],
        }

        project_info = self.get_info_by_id(id)

        custom_data = project_info.custom_data or {}
        if "import_history" not in custom_data:
            custom_data["import_history"] = {"tasks": []}
        if "tasks" not in custom_data["import_history"]:
            custom_data["import_history"]["tasks"] = []
        custom_data["import_history"]["tasks"].append(data)

        self.edit_info(id, custom_data=custom_data)
