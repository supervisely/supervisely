# coding: utf-8

from enum import Enum
import urllib
from collections import defaultdict
from typing import Dict

from supervisely.api.module_api import (
    ApiField,
    CloneableModuleApi,
    UpdateableModule,
    RemoveableModuleApi,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.annotation.annotation import TagCollection


class ProjectNotFound(Exception):
    pass


class ExpectedProjectTypeMismatch(Exception):
    pass


class ProjectApi(CloneableModuleApi, UpdateableModule, RemoveableModuleApi):
    @staticmethod
    def info_sequence():
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
        ]

    @staticmethod
    def info_tuple_name():
        return "ProjectInfo"

    def __init__(self, api):
        CloneableModuleApi.__init__(self, api)
        UpdateableModule.__init__(self, api)

    def get_list(self, workspace_id, filters=None):
        """
        :param workspace_id: int
        :param filters: list
        :return: list all the projects for a given workspace
        """
        return self.get_list_all_pages(
            "projects.list",
            {ApiField.WORKSPACE_ID: workspace_id, "filter": filters or []},
        )

    def get_info_by_id(self, id, expected_type=None, raise_error=False):
        """
        :param id: int
        :param expected_type: type of data we expext to get info (raise error if type of project is not None and != expected type)
        :param raise_error: bool
        :return: project metadata by numeric id (None if request status_code == 404, raise error in over way)
        """
        info = self._get_info_by_id(id, "projects.info")
        self._check_project_info(
            info, id=id, expected_type=expected_type, raise_error=raise_error
        )
        return info

    def get_info_by_name(self, parent_id, name, expected_type=None, raise_error=False):
        """
        :param parent_id: int
        :param name: str
        :param expected_type: type of data we expext to get info
        :param raise_error: bool
        :return: project metadata by numeric workspace id and given name of project
        """
        info = super().get_info_by_name(parent_id, name)
        self._check_project_info(
            info, name=name, expected_type=expected_type, raise_error=raise_error
        )
        return info

    def _check_project_info(
        self, info, id=None, name=None, expected_type=None, raise_error=False
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

    def get_meta(self, id):
        """
        :param id: int
        :return: labeling meta information for the project - the set of available object classes and tags
        """
        response = self._api.post("projects.meta", {"id": id})
        return response.json()

    def clone_advanced(self, 
                       id, 
                       dst_workspace_id, 
                       dst_name,
                       with_meta=True,
                       with_datasets=True,
                       with_items=True,
                       with_annotations=True):
        if not with_meta and with_annotations:
            raise ValueError("with_meta parameter must be True if with_annotations parameter is True")
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
                    ApiField.FIGURES_TAGS: with_annotations
                },

            },
        )
        return response.json()[ApiField.TASK_ID]

    def create(
        self,
        workspace_id,
        name,
        type=ProjectType.IMAGES,
        description="",
        change_name_if_conflict=False,
    ):
        """
        Create project with given name in workspace with given id
        :param workspace_id: int
        :param name: str
        :param type: type of progect to create
        :param description: str
        :param change_name_if_conflict: bool
        :return: created project metadata
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
        return "projects.editInfo"

    def update_meta(self, id, meta):
        """
        Update given project with given metadata
        :param id: int
        :param meta: project metainformation
        """
        self._api.post("projects.meta.update", {ApiField.ID: id, ApiField.META: meta})

    def _clone_api_method_name(self):
        return "projects.clone"

    def get_datasets_count(self, id):
        """
        :param id: int
        :return: int (number of datasets in given project)
        """
        datasets = self._api.dataset.get_list(id)
        return len(datasets)

    def get_images_count(self, id):
        """
        :param id: int
        :return: int (number of images in given project)
        """
        datasets = self._api.dataset.get_list(id)
        return sum([dataset.images_count for dataset in datasets])

    def _remove_api_method_name(self):
        return "projects.remove"

    def merge_metas(self, src_project_id, dst_project_id):
        """
        Add metadata from given progect to given destination project
        :param src_project_id: int
        :param dst_project_id: int
        :return: merged project metainformation
        """
        if src_project_id == dst_project_id:
            return self.get_meta(src_project_id)

        src_meta = ProjectMeta.from_json(self.get_meta(src_project_id))
        dst_meta = ProjectMeta.from_json(self.get_meta(dst_project_id))

        new_dst_meta = src_meta.merge(dst_meta)
        new_dst_meta_json = new_dst_meta.to_json()
        self.update_meta(dst_project_id, new_dst_meta.to_json())

        return new_dst_meta_json

    def get_activity(self, id, progress_cb=None):
        import pandas as pd

        proj_info = self.get_info_by_id(id)
        workspace_info = self._api.workspace.get_info_by_id(proj_info.workspace_id)
        activity = self._api.team.get_activity(
            workspace_info.team_id, filter_project_id=id, progress_cb=progress_cb
        )
        df = pd.DataFrame(activity)
        return df

    def _convert_json_info(self, info: dict, skip_missing=True):
        res = super()._convert_json_info(info, skip_missing=skip_missing)
        if res.reference_image_url is not None:
            res = res._replace(
                reference_image_url=res.reference_image_url
                )
        if res.items_count is None:
            res = res._replace(items_count=res.images_count)
        return res

    def get_stats(self, id):
        response = self._api.post("projects.stats", {ApiField.ID: id})
        return response.json()

    def url(self, id):
        return f"projects/{id}/datasets"

    def update_custom_data(self, id, data):
        if type(data) is not dict:
            raise TypeError("Meta must be dict, not {!r}".format(type(data)))
        response = self._api.post(
            "projects.editInfo", {ApiField.ID: id, ApiField.CUSTOM_DATA: data}
        )
        return response.json()

    def update_settings(self, id: int, settings: Dict[str, str]) -> None:
        """
        Updates project wuth given project settings by id.

        :param id: Project ID
        :type id: int
        :param settings: Project settings
        :type settings: Dict[str, str]
        """
        self._api.post(
            "projects.settings.update", {ApiField.ID: id, ApiField.SETTINGS: settings}
        )

    def download_images_tags(self, id, progress_cb=None):
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
        """
        Enables images grouping in project.

        :param id: Project ID
        :type id: int
        :param enable: if True groups images by given tag name
        :type enable: Dict[str, str]
        :param tag_name: Name of the tag. Images will be grouped by this tag
        :type tag_name: str
        """
        project_meta_json = self.get_meta(id)
        project_meta = ProjectMeta.from_json(project_meta_json)
        group_tag_meta = project_meta.get_tag_meta(tag_name)
        if group_tag_meta is None:
            raise Exception(f"Tag {tag_name} doesn't exists in the given project")

        group_tag_id = group_tag_meta.sly_id
        project_settings = {"groupImages": enable, "groupImagesByTagId": group_tag_id, "groupImagesSync": sync}
        self.update_settings(id=id, settings=project_settings)

    def get_or_create(
        self, workspace_id, name, type=ProjectType.IMAGES, description=""
    ):
        info = self.get_info_by_name(workspace_id, name)
        if info is None:
            info = self.create(workspace_id, name, type=type, description=description)
        return info

    def edit_info(self, id, name=None, description=None, readme=None, custom_data=None, project_type=None):
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
            if not (current_project_type == str(ProjectType.POINT_CLOUDS) and project_type == str(ProjectType.POINT_CLOUD_EPISODES)):
                raise ValueError(f"conversion from {current_project_type} to {project_type} is not supported ")
            body[ApiField.TYPE] = project_type

        response = self._api.post(self._get_update_method(), body)
        return self._convert_json_info(response.json())
