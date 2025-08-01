# coding: utf-8
"""create or manipulate already existing labeling jobs"""

# docs
from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
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

import requests

from supervisely.annotation.annotation import Annotation
from supervisely.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely.annotation.label import Label
from supervisely.annotation.tag import Tag
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.module_api import (
    ApiField,
    ModuleApi,
    ModuleWithStatus,
    RemoveableBulkModuleApi,
    WaitingTimeExceeded,
)
from supervisely.collection.str_enum import StrEnum
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.graph import GraphNodes
from supervisely.geometry.point import Point
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger


class LabelingJobInfo(NamedTuple):
    id: int
    name: str
    readme: str
    description: str
    team_id: int
    workspace_id: int
    workspace_name: str
    project_id: int
    project_name: str
    dataset_id: int
    dataset_name: str
    created_by_id: int
    created_by_login: str
    assigned_to_id: int
    assigned_to_login: str
    reviewer_id: int
    reviewer_login: str
    created_at: str
    started_at: str
    finished_at: str
    status: str
    disabled: bool
    labeling_queue_id: int
    labeling_exam_id: int
    images_count: int
    finished_images_count: int
    rejected_images_count: int
    accepted_images_count: int
    progress_images_count: int
    classes_to_label: list
    tags_to_label: list
    images_range: tuple
    objects_limit_per_image: int
    tags_limit_per_image: int
    filter_images_by_tags: list
    include_images_with_tags: list
    exclude_images_with_tags: list
    entities: list
    priority: int


class LabelingJobApi(RemoveableBulkModuleApi, ModuleWithStatus):
    """
    API for working with Labeling Jobs. :class:`LabelingJobApi<LabelingJobApi>` object is immutable.

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

        jobs = api.labeling_job.get_list(9) # api usage example
    """

    class Status(StrEnum):
        """Labeling Job status."""

        PENDING = "pending"
        """"""
        IN_PROGRESS = "in_progress"
        """"""
        ON_REVIEW = "on_review"
        """"""
        COMPLETED = "completed"
        """"""
        STOPPED = "stopped"
        """"""
        REVIEW_COMPLETED = "review_completed"
        """"""

    @staticmethod
    def info_sequence():
        """
        NamedTuple LabelingJobInfo information about Labeling Job.

        :Example:

         .. code-block:: python

             LabelingJobInfo(id=2,
                             name='Annotation Job (#1) (#1) (dataset_01)',
                             readme='',
                             description='',
                             team_id=4,
                             workspace_id=8,
                             workspace_name='First Workspace',
                             project_id=58,
                             project_name='tutorial_project',
                             dataset_id=54,
                             dataset_name='dataset_01',
                             created_by_id=4,
                             created_by_login='anna',
                             assigned_to_id=4,
                             assigned_to_login='anna',
                             reviewer_id=4,
                             reviewer_login='anna',
                             created_at='2020-04-08T15:10:12.618Z',
                             started_at='2020-04-08T15:10:19.833Z',
                             finished_at='2020-04-08T15:13:39.788Z',
                             status='completed',
                             disabled=False,
                             labeling_queue_id=3,
                             labeling_exam_id=None,
                             images_count=3,
                             finished_images_count=0,
                             rejected_images_count=1,
                             accepted_images_count=2,
                             progress_images_count=2,
                             classes_to_label=[],
                             tags_to_label=[],
                             images_range=(1, 5),
                             objects_limit_per_image=None,
                             tags_limit_per_image=None,
                             filter_images_by_tags=[],
                             include_images_with_tags=[],
                             exclude_images_with_tags=[],
                             entities=None,
                             priority=2)
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.README,
            ApiField.DESCRIPTION,
            ApiField.TEAM_ID,
            ApiField.WORKSPACE_ID,
            ApiField.WORKSPACE_NAME,
            ApiField.PROJECT_ID,
            ApiField.PROJECT_NAME,
            ApiField.DATASET_ID,
            ApiField.DATASET_NAME,
            ApiField.CREATED_BY_ID,
            ApiField.CREATED_BY_LOGIN,
            ApiField.ASSIGNED_TO_ID,
            ApiField.ASSIGNED_TO_LOGIN,
            ApiField.REVIEWER_ID,
            ApiField.REVIEWER_LOGIN,
            ApiField.CREATED_AT,
            ApiField.STARTED_AT,
            ApiField.FINISHED_AT,
            ApiField.STATUS,
            ApiField.DISABLED,
            ApiField.LABELING_QUEUE_ID,
            ApiField.LABELING_EXAM_ID,
            ApiField.IMAGES_COUNT,
            ApiField.FINISHED_IMAGES_COUNT,
            ApiField.REJECTED_IMAGES_COUNT,
            ApiField.ACCEPTED_IMAGES_COUNT,
            ApiField.PROGRESS_IMAGES_COUNT,
            ApiField.CLASSES_TO_LABEL,
            ApiField.TAGS_TO_LABEL,
            ApiField.IMAGES_RANGE,
            ApiField.OBJECTS_LIMIT_PER_IMAGE,
            ApiField.TAGS_LIMIT_PER_IMAGE,
            ApiField.FILTER_IMAGES_BY_TAGS,
            ApiField.INCLUDE_IMAGES_WITH_TAGS,
            ApiField.EXCLUDE_IMAGES_WITH_TAGS,
            ApiField.ENTITIES,
            ApiField.PRIORITY,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **LabelingJobInfo**.
        """
        return "LabelingJobInfo"

    def __init__(self, api):
        ModuleApi.__init__(self, api)

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
        """ """
        if info is None:
            return None
        else:
            field_values = []
            for field_name in self.info_sequence():
                if field_name in [
                    ApiField.INCLUDE_IMAGES_WITH_TAGS,
                    ApiField.EXCLUDE_IMAGES_WITH_TAGS,
                ]:
                    continue
                value = None
                if type(field_name) is str:
                    if skip_missing is True:
                        value = info.get(field_name, None)
                    else:
                        value = info[field_name]
                elif type(field_name) is tuple:
                    for sub_name in field_name[0]:
                        if value is None:
                            if skip_missing is True:
                                value = info.get(sub_name, None)
                            else:
                                value = info[sub_name]
                        else:
                            value = value[sub_name]
                else:
                    raise RuntimeError("Can not parse field {!r}".format(field_name))

                if field_name == ApiField.FILTER_IMAGES_BY_TAGS:
                    field_values.append(value)
                    include_images_with_tags = []
                    exclude_images_with_tags = []
                    for fv in value:
                        key = ApiField.NAME
                        if key not in fv:
                            key = "title"
                        if fv["positive"] is True:
                            include_images_with_tags.append(fv[key])
                        else:
                            exclude_images_with_tags.append(fv[key])
                    field_values.append(include_images_with_tags)
                    field_values.append(exclude_images_with_tags)
                    continue
                elif (
                    field_name == ApiField.CLASSES_TO_LABEL or field_name == ApiField.TAGS_TO_LABEL
                ):
                    new_value = []
                    for fv in value:
                        key = ApiField.NAME
                        if ApiField.NAME not in fv:
                            key = "title"
                        new_value.append(fv[key])
                    field_values.append(new_value)
                    continue
                elif field_name == ApiField.IMAGES_RANGE:
                    value = (value["start"], value["end"])

                field_values.append(value)

            res = self.InfoType(*field_values)
            return LabelingJobInfo(**res._asdict())

    def _remove_batch_api_method_name(self):
        """Api remove method name."""

        return "jobs.bulk.remove"

    def _remove_batch_field_name(self):
        """Returns onstant string that represents API field name."""

        return ApiField.IDS

    def _check_membership(self, ids: List[int], team_id: int) -> None:
        """Check if user is a member of the team in which the Labeling Job is created."""
        for user_id in ids:
            memberships = self._api.user.get_teams(user_id)
            team_ids = [team.id for team in memberships]
            if team_id not in team_ids:
                raise RuntimeError(
                    f"User with id {user_id} is not a member of the team with id {team_id}"
                )

    def create(
        self,
        name: str,
        dataset_id: int,
        user_ids: List[int],
        readme: Optional[str] = None,
        description: Optional[str] = None,
        classes_to_label: Optional[List[str]] = None,
        objects_limit_per_image: Optional[int] = None,
        tags_to_label: Optional[List[str]] = None,
        tags_limit_per_image: Optional[int] = None,
        include_images_with_tags: Optional[List[str]] = None,
        exclude_images_with_tags: Optional[List[str]] = None,
        images_range: Optional[List[int, int]] = None,
        reviewer_id: Optional[int] = None,
        images_ids: Optional[List[int]] = [],
        dynamic_classes: Optional[bool] = False,
        dynamic_tags: Optional[bool] = False,
        disable_confirm: Optional[bool] = None,
        disable_submit: Optional[bool] = None,
        toolbox_settings: Optional[Dict] = None,
        enable_quality_check: Optional[bool] = None,
    ) -> List[LabelingJobInfo]:
        """
        Creates Labeling Job and assigns given Users to it.

        :param name: Labeling Job name in Supervisely.
        :type name: str
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param user_ids: User IDs in Supervisely to assign Users as labelers to Labeling Job.
        :type user_ids: List[int]
        :param readme: Additional information about Labeling Job.
        :type readme: str, optional
        :param description: Description of Labeling Job.
        :type description: str, optional
        :param classes_to_label: List of classes to label in Dataset.
        :type classes_to_label: List[str], optional
        :param objects_limit_per_image: Limit the number of objects that the labeler can create on each image.
        :type objects_limit_per_image: int, optional
        :param tags_to_label: List of tags to label in Dataset.
        :type tags_to_label: List[str], optional
        :param tags_limit_per_image: Limit the number of tags that the labeler can create on each image.
        :type tags_limit_per_image: int, optional
        :param include_images_with_tags: Include images with given tags for processing by labeler.
        :type include_images_with_tags: List[str], optional
        :param exclude_images_with_tags: Exclude images with given tags for processing by labeler.
        :type exclude_images_with_tags: List[str], optional
        :param images_range: Limit number of images to be labeled for each labeler.
        :type images_range: List[int, int], optional
        :param reviewer_id: User ID in Supervisely to assign User as Reviewer to Labeling Job.
        :type reviewer_id: int, optional
        :param images_ids: List of images ids to label in dataset
        :type images_ids: List[int], optional
        :param dynamic_classes: If True, classes created after creating the job will be available for annotators
        :type dynamic_classes: bool, optional
        :param dynamic_tags: If True, tags created after creating the job will be available for annotators
        :type dynamic_tags: bool, optional
        :param disable_confirm: If True, the Confirm button will be disabled in the labeling tool. It will remain disabled until the next API call sets the parameter to False, re-enabling the button.
        :type disable_confirm: bool, optional
        :param disable_submit: If True, the Submit button will be disabled in the labeling tool. It will remain disabled until the next API call sets the parameter to False, re-enabling the button.
        :type disable_submit: bool, optional
        :param toolbox_settings: Settings for the labeling tool. Only video projects are supported.
        :type toolbox_settings: Dict, optional
        :param enable_quality_check: If True, adds an intermediate step between "review" and completing the Labeling Job.
        :type enable_quality_check: bool, optional
        :return: List of information about new Labeling Job. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[LabelingJobInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_name = 'alex'
            dataset_id = 602
            new_labeling_jobs = api.labeling_job.create(
                user_name,
                dataset_id,
                user_ids=[111, 222],
                readme='Readmy text',
                description='Work for labelers',
                objects_limit_per_image=5,
                tags_limit_per_image=3
            )
            print(new_labeling_jobs)

            # >>> List[LabelingJobInfo(id=2,...)]

            # Create video labeling job with toolbox settings

            user_id = 4
            dataset_id = 277
            video_id = 24897
            toolbox_settings = {"playbackRate": 32, "skipFramesSize": 15, "showVideoTime": True}

            new_labeling_jobs = api.labeling_job.create(
                name="Labeling Job name",
                dataset_id=dataset_id,
                user_ids=[user_id],
                readme="Labeling Job readme",
                description="Some description",
                classes_to_label=["car", "animal"],
                tags_to_label=["animal_age_group"],
                images_ids=[video_id],
                toolbox_settings=toolbox_settings,
            )
            print(new_labeling_jobs)

            # >>> List[LabelingJobInfo(id=3,...)]
        """
        if classes_to_label is None:
            classes_to_label = []
        if tags_to_label is None:
            tags_to_label = []

        filter_images_by_tags = []
        if include_images_with_tags is not None:
            for tag_name in include_images_with_tags:
                filter_images_by_tags.append({"name": tag_name, "positive": True})

        if exclude_images_with_tags is not None:
            for tag_name in exclude_images_with_tags:
                filter_images_by_tags.append({"name": tag_name, "positive": False})

        if objects_limit_per_image is None:
            objects_limit_per_image = 0

        if tags_limit_per_image is None:
            tags_limit_per_image = 0

        meta = {
            "classes": classes_to_label,
            "projectTags": tags_to_label,
            "imageTags": filter_images_by_tags,
            "imageFiguresLimit": objects_limit_per_image,
            "imageTagsLimit": tags_limit_per_image,
            "entityIds": images_ids,
            "dynamicClasses": dynamic_classes,
            "dynamicTags": dynamic_tags,
        }

        if toolbox_settings is not None:
            dataset_info = self._api.dataset.get_info_by_id(dataset_id)
            project_id = dataset_info.project_id
            project_info = self._api.project.get_info_by_id(project_id)
            project_type = project_info.type
            if project_type == ProjectType.VIDEOS.value:
                playback_rate_possible_values = [
                    0.1,
                    0.3,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                    1.1,
                    1.2,
                    1.3,
                    1.5,
                    2,
                    4,
                    8,
                    16,
                    32,
                ]
                playback_rate = toolbox_settings.get("playbackRate", None)
                if playback_rate is not None:
                    if playback_rate not in playback_rate_possible_values:
                        raise ValueError(
                            f"'playbackRate' must be one of: '{','.join(playback_rate_possible_values)}'"
                        )
                meta["toolboxSettings"] = toolbox_settings

        if disable_confirm is not None:
            meta.update({"disableConfirm": disable_confirm})
        if disable_submit is not None:
            meta.update({"disableSubmit": disable_submit})
        if enable_quality_check is not None:
            meta.update({"enableIntermediateReview": enable_quality_check})

        data = {
            ApiField.NAME: name,
            ApiField.DATASET_ID: dataset_id,
            ApiField.USER_IDS: user_ids,
            # ApiField.DESCRIPTION: description,
            ApiField.META: meta,
        }

        if readme is not None:
            data[ApiField.README] = str(readme)

        if description is not None:
            data[ApiField.DESCRIPTION] = str(description)

        if images_range is not None and images_range != (None, None):
            if len(images_range) != 2:
                raise RuntimeError("images_range has to contain 2 elements (start, end)")
            images_range = {"start": images_range[0], "end": images_range[1]}
            data[ApiField.META]["range"] = images_range

        if reviewer_id is not None:
            data[ApiField.REVIEWER_ID] = reviewer_id

        response = self._api.post("jobs.add", data)
        # created_jobs_json = response.json()

        created_jobs = []
        for job in response.json():
            created_jobs.append(self.get_info_by_id(job[ApiField.ID]))
        return created_jobs

    def get_list(
        self,
        team_id: int,
        created_by_id: Optional[int] = None,
        assigned_to_id: Optional[int] = None,
        project_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        show_disabled: Optional[bool] = False,
        reviewer_id: Optional[int] = None,
        is_part_of_queue: Optional[bool] = True,
        queue_ids: Optional[Union[List, int]] = None,
        exclude_statuses: Optional[
            List[Literal["pending", "in_progress", "on_review", "completed"]]
        ] = None,
    ) -> List[LabelingJobInfo]:
        """
        Get list of information about Labeling Job in the given Team.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param created_by_id: ID of the User who created the LabelingJob.
        :type created_by_id: int, optional
        :param assigned_to_id: ID of the assigned User.
        :type assigned_to_id: int, optional
        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int, optional
        :param show_disabled: Show disabled Labeling Jobs.
        :type show_disabled: bool, optional
        :param reviewer_id: ID of the User who reviews the LabelingJob.
        :type reviewer_id: int, optional
        :param is_part_of_queue: Filter by Labeling Queue. If True, all existing Labeling Jobs are returned. If False, only Labeling Jobs that are not part of the queue are returned.
        :type is_part_of_queue: bool, optional
        :param queue_ids: IDs of the Labeling Queues. If set, only Labeling Jobs from the selected queues are returned. Arg `is_part_of_queue` must be True.
        :type queue_ids: Union[List, int], optional
        :param exclude_statuses: Exclude Labeling Jobs with given statuses.
        :type exclude_statuses: List[Literal["pending", "in_progress", "on_review", "completed"]], optional
        :return: List of information about Labeling Jobs. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[LabelingJobInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            label_jobs = api.labeling_job.get_list(4)
            print(label_jobs)
            # Output: [
            #     [
            #         2,
            #         "Annotation Job (#1) (#1) (dataset_01)",
            #         "",
            #         "",
            #         4,
            #         8,
            #         "First Workspace",
            #         58,
            #         "tutorial_project",
            #         54,
            #         "dataset_01",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         "2020-04-08T15:10:12.618Z",
            #         "2020-04-08T15:10:19.833Z",
            #         "2020-04-08T15:13:39.788Z",
            #         "completed",
            #         false,
            #         3,
            #         null,
            #         3,
            #         0,
            #         1,
            #         2,
            #         2,
            #         [],
            #         [],
            #         [
            #             1,
            #             5
            #         ],
            #         null,
            #         null,
            #         [],
            #         [],
            #         [],
            #         null
            #     ],
            #     [
            #         3,
            #         "Annotation Job (#1) (#2) (dataset_02)",
            #         "",
            #         "",
            #         4,
            #         8,
            #         "First Workspace",
            #         58,
            #         "tutorial_project",
            #         55,
            #         "dataset_02",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         4,
            #         "anna",
            #         "2020-04-08T15:10:12.618Z",
            #         "2020-04-08T15:15:46.749Z",
            #         "2020-04-08T15:17:33.572Z",
            #         "completed",
            #         false,
            #         3,
            #         null,
            #         2,
            #         0,
            #         0,
            #         2,
            #         2,
            #         [],
            #         [],
            #         [
            #             1,
            #             5
            #         ],
            #         null,
            #         null,
            #         [],
            #         [],
            #         [],
            #         null
            #     ]
            # ]
        """
        if not is_part_of_queue and queue_ids is not None:
            raise ValueError("To filter by `queue_id`, `is_part_of_queue` must be set to `True`.")

        if isinstance(queue_ids, int):
            queue_ids = [queue_ids]

        filters = []
        if created_by_id is not None:
            filters.append(
                {"field": ApiField.CREATED_BY_ID[0][0], "operator": "=", "value": created_by_id}
            )
        if assigned_to_id is not None:
            filters.append(
                {"field": ApiField.ASSIGNED_TO_ID[0][0], "operator": "=", "value": assigned_to_id}
            )
        if reviewer_id is not None:
            filters.append({"field": ApiField.REVIEWER_ID, "operator": "=", "value": reviewer_id})
        if project_id is not None:
            filters.append({"field": ApiField.PROJECT_ID, "operator": "=", "value": project_id})
        if dataset_id is not None:
            filters.append({"field": ApiField.DATASET_ID, "operator": "=", "value": dataset_id})
        if not is_part_of_queue:
            filters.append({"field": ApiField.LABELING_QUEUE_ID, "operator": "=", "value": None})
        if queue_ids is not None:
            filters.append(
                {"field": ApiField.LABELING_QUEUE_ID, "operator": "in", "value": queue_ids}
            )
        if exclude_statuses is not None:
            filters.append({"field": ApiField.STATUS, "operator": "!in", "value": exclude_statuses})
        return self.get_list_all_pages(
            "jobs.list",
            {ApiField.TEAM_ID: team_id, "showDisabled": show_disabled, ApiField.FILTER: filters},
        )

    def stop(self, id: int) -> None:
        """
        Makes Labeling Job unavailable for labeler with given User ID.

        :param id: User ID in Supervisely.
        :type id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.stop(9)
        """
        self._api.post("jobs.stop", {ApiField.ID: id})

    def get_info_by_id(self, id: int) -> LabelingJobInfo:
        """
        Get Labeling Job information by ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :return: Information about Labeling Job. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`LabelingJobInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            label_job_info = api.labeling_job.get_info_by_id(2)
            print(label_job_info)
            # Output: [
            #     2,
            #     "Annotation Job (#1) (#1) (dataset_01)",
            #     "",
            #     "",
            #     4,
            #     8,
            #     "First Workspace",
            #     58,
            #     "tutorial_project",
            #     54,
            #     "dataset_01",
            #     4,
            #     "anna",
            #     4,
            #     "anna",
            #     4,
            #     "anna",
            #     "2020-04-08T15:10:12.618Z",
            #     "2020-04-08T15:10:19.833Z",
            #     "2020-04-08T15:13:39.788Z",
            #     "completed",
            #     false,
            #     3,
            #     0,
            #     1,
            #     2,
            #     2,
            #     [],
            #     [],
            #     [
            #         1,
            #         5
            #     ],
            #     null,
            #     null,
            #     [],
            #     [],
            #     [],
            #     [
            #         {
            #             "reviewStatus": "rejected",
            #             "id": 283,
            #             "name": "image_03"
            #         },
            #         {
            #             "reviewStatus": "accepted",
            #             "id": 282,
            #             "name": "image_02"
            #         },
            #         {
            #             "reviewStatus": "accepted",
            #             "id": 281,
            #             "name": "image_01"
            #         }
            #     ]
            # ]
        """
        return self._get_info_by_id(id, "jobs.info")

    def archive(self, id: int) -> None:
        """
        Archives Labeling Job with given ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.archive(23)
        """
        self._api.post("jobs.archive", {ApiField.ID: id})

    def get_status(self, id: int) -> LabelingJobApi.Status:
        """
        Get status of Labeling Job with given ID.

        :param id: Labeling job ID in Supervisely.
        :type id: int
        :return: Labeling Job Status
        :rtype: :class:`Status<supervisely.api.labeling_job_api.LabelingJobApi.Status>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            job_status = api.labeling_job.get_status(4)
            print(job_status) # pending
        """
        status_str = self.get_info_by_id(id).status
        return self.Status(status_str)

    def raise_for_status(self, status):
        """ """
        # there is no ERROR status for labeling job
        pass

    def wait(
        self,
        id: int,
        target_status: str,
        wait_attempts: Optional[int] = None,
        wait_attempt_timeout_sec: Optional[int] = None,
    ) -> None:
        """
        Wait for a Labeling Job to change to the expected target status.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param target_status: Expected result status of Labeling Job.
        :type target_status: str
        :param wait_attempts: Number of attempts to retry, when :class:`WaitingTimeExceeded` raises.
        :type wait_attempts: int, optional
        :param wait_attempt_timeout_sec: Time between attempts.
        :type wait_attempt_timeout_sec: int, optional
        :raises: :class:`WaitingTimeExceeded`, if waiting time exceeded
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.wait(4, 'completed', wait_attempts=2, wait_attempt_timeout_sec=1)
            # supervisely.api.module_api.WaitingTimeExceeded: Waiting time exceeded
        """
        wait_attempts = wait_attempts or self.MAX_WAIT_ATTEMPTS
        effective_wait_timeout = wait_attempt_timeout_sec or self.WAIT_ATTEMPT_TIMEOUT_SEC
        for attempt in range(wait_attempts):
            status = self.get_status(id)
            self.raise_for_status(status)
            if status is target_status:
                return
            time.sleep(effective_wait_timeout)
        raise WaitingTimeExceeded("Waiting time exceeded")

    def get_stats(self, id: int) -> Dict:
        """
        Get stats of given Labeling Job ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :return: Dict with information about given Labeling Job
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            status = api.labeling_job.get_stats(3)
            print(status)
            # Output: {
            #     "job": {
            #         "editingDuration": 0,
            #         "annotationDuration": 720,
            #         "id": 3,
            #         "name": "Annotation Job (#1) (#2) (dataset_02)",
            #         "startedAt": "2020-04-08T15:15:46.749Z",
            #         "finishedAt": "2020-04-08T15:17:33.572Z",
            #         "imagesCount": 2,
            #         "finishedImagesCount": 2,
            #         "tagsStats": [
            #             {
            #                 "id": 24,
            #                 "color": "#ED68A1",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "car_color"
            #             },
            #             {
            #                 "id": 19,
            #                 "color": "#A0A08C",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "cars_number"
            #             },
            #             {
            #                 "id": 20,
            #                 "color": "#D98F7E",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "like"
            #             },
            #             {
            #                 "id": 23,
            #                 "color": "#65D37C",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "person_gender"
            #             },
            #             {
            #                 "parentId": 23,
            #                 "color": "#65D37C",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "person_gender (male)"
            #             },
            #             {
            #                 "parentId": 23,
            #                 "color": "#65D37C",
            #                 "images": 0,
            #                 "figures": 0,
            #                 "name": "person_gender (female)"
            #             },
            #             {
            #                 "id": 21,
            #                 "color": "#855D79",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "situated"
            #             },
            #             {
            #                 "parentId": 21,
            #                 "color": "#855D79",
            #                 "images": 1,
            #                 "figures": 1,
            #                 "name": "situated (inside)"
            #             },
            #             {
            #                 "parentId": 21,
            #                 "color": "#855D79",
            #                 "images": 0,
            #                 "figures": 0,
            #                 "name": "situated (outside)"
            #             },
            #             {
            #                 "id": 22,
            #                 "color": "#A2B4FA",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "vehicle_age"
            #             },
            #             {
            #                 "parentId": 22,
            #                 "color": "#A2B4FA",
            #                 "images": 0,
            #                 "figures": 1,
            #                 "name": "vehicle_age (modern)"
            #             },
            #             {
            #                 "parentId": 22,
            #                 "color": "#A2B4FA",
            #                 "images": 0,
            #                 "figures": 0,
            #                 "name": "vehicle_age (vintage)"
            #             }
            #         ]
            #     },
            #     "classes": [
            #         {
            #             "id": 43,
            #             "color": "#F6FF00",
            #             "shape": "rectangle",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "bike",
            #             "labelsCount": 0
            #         },
            #         {
            #             "id": 42,
            #             "color": "#BE55CE",
            #             "shape": "polygon",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "car",
            #             "labelsCount": 0
            #         },
            #         {
            #             "id": 41,
            #             "color": "#FD0000",
            #             "shape": "polygon",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "dog",
            #             "labelsCount": 0
            #         },
            #         {
            #             "id": 40,
            #             "color": "#00FF12",
            #             "shape": "bitmap",
            #             "totalDuration": 0,
            #             "imagesCount": 0,
            #             "avgDuration": null,
            #             "name": "person",
            #             "labelsCount": 0
            #         }
            #     ],
            #     "images": {
            #         "total": 2,
            #         "images": [
            #             {
            #                 "id": 285,
            #                 "reviewStatus": "accepted",
            #                 "annotationDuration": 0,
            #                 "totalDuration": 0,
            #                 "name": "image_01",
            #                 "labelsCount": 0
            #             },
            #             {
            #                 "id": 284,
            #                 "reviewStatus": "accepted",
            #                 "annotationDuration": 0,
            #                 "totalDuration": 0,
            #                 "name": "image_02",
            #                 "labelsCount": 0
            #             }
            #         ]
            #     }
            # }
        """
        response = self._api.post("jobs.stats", {ApiField.ID: id})
        return response.json()

    def get_activity(
        self, team_id: int, job_id: int, progress_cb: Optional[Union[tqdm, Callable]] = None
    ) -> DataFrame:
        import pandas as pd

        """
        Get all activity for given Labeling Job by ID.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param job_id: Labeling Job ID in Supervisely.
        :type job_id: int
        :param progress_cb: Function for tracking progress
        :type progress_cb: tqdm, optional
        :return: Activity data as `pd.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        :rtype: :class:`pd.DataFrame`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            activity = api.labeling_job.get_activity(3)
            print(activity)
            # Output:
            #   userId         action  ... tagId                 meta
            # 0       4  update_figure  ...   NaN                   {}
            # 1       4  create_figure  ...   NaN                   {}
            # 2       4     attach_tag  ...  20.0                   {}
            # 3       4     attach_tag  ...  21.0  {'value': 'inside'}
            # 4       4     attach_tag  ...  24.0      {'value': '12'}
            # 5       4  update_figure  ...   NaN                   {}
            # 6       4  update_figure  ...   NaN                   {}
            # 7       4  update_figure  ...   NaN                   {}
            # 8       4  create_figure  ...   NaN                   {}
            # 9       4  update_figure  ...   NaN                   {}
            # [10 rows x 18 columns]
        """
        activity = self._api.team.get_activity(
            team_id, filter_job_id=job_id, progress_cb=progress_cb
        )
        df = pd.DataFrame(activity)
        return df

    def set_status(self, id: int, status: str) -> None:
        """
        Sets Labeling Job status.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param status: New Labeling Job status
        :type status: str
        :return: None
        :rtype: :class:`NoneType`
        :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely.api.labeling_job_api.LabelingJobApi.Status import COMPLETED

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            api.labeling_job.set_status(id=9, status="completed")
        """
        self._api.post("jobs.set-status", {ApiField.ID: id, ApiField.STATUS: status})

    def get_project_meta(self, id: int) -> ProjectMeta:
        """
        Returns project meta with classes and tags used in the labeling job with given id.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :return: Project meta of the labeling job with given id.
        :rtype: :class:`ProjectMeta`
        """
        job_info = self.get_info_by_id(id)
        project_meta_json = self._api.project.get_meta(job_info.project_id)
        project_meta = ProjectMeta.from_json(project_meta_json)

        job_classes = [
            obj_class
            for obj_class in project_meta.obj_classes
            if obj_class.name in job_info.classes_to_label
        ]
        job_tags = [
            tag_meta
            for tag_meta in project_meta.tag_metas
            if tag_meta.name in job_info.tags_to_label
        ]
        job_meta = ProjectMeta(obj_classes=job_classes, tag_metas=job_tags)
        return job_meta

    def get_annotations(
        self,
        id: int,
        image_ids: Optional[List[int]] = None,
        project_meta: Optional[ProjectMeta] = None,
        image_infos: Optional[List[ImageInfo]] = None,
    ) -> List[Annotation]:
        """
        Return annotations for given image ids from labeling job with given id.
        To speed up the process, you can provide image infos, which will be used instead of fetching them from the API.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param image_ids: Image IDs in Supervisely.
                        If not provided, you must provide :param:`image_infos`.
                        Have lower priority than :param:`image_infos`.
        :type image_ids: List[int], optional
        :param project_meta: Project meta of the labeling job with given id. Can be retrieved with :func:`get_project_meta`.
        :type project_meta: :class:`ProjectMeta`, optional
        :param image_infos: List of ImageInfo objects.
                            If not provided, will be retrieved from the API.
                            Have higher priority than :param:`image_ids`.
        :type image_infos: List[ImageInfo], optional
        :return: Annotation for given image id from labeling job with given id.
        :rtype: :class:`Annotation`
        """

        def _get_geometry(type: str, data: dict):
            try:
                geometry_type = GET_GEOMETRY_FROM_STR(type)
                geometry = geometry_type.from_json(data)
            except Exception as e:
                logger.error(f"Can't parse geometry: {repr(e)}")
                geometry = None
            return geometry

        def _create_tags_from_labeling_job(
            job_tags_map: List[dict], project_meta: ProjectMeta
        ) -> List[Tag]:
            tags = []
            for tag in job_tags_map:
                tag_meta = project_meta.get_tag_meta_by_id(tag["tagId"])
                if tag_meta is None:
                    continue
                tag_value = tag.get("value", None)
                tag = Tag(tag_meta, value=tag_value)
                tags.append(tag)
            return tags

        def _create_labels_from_labeling_job(
            figures_map: List[FigureInfo], project_meta: ProjectMeta
        ) -> List[Label]:
            labels = []
            for figure in figures_map:
                obj_class = project_meta.get_obj_class_by_id(figure.class_id)
                if obj_class is None:
                    continue
                geometry = _get_geometry(figure.geometry_type, figure.geometry)
                if geometry is None:
                    continue
                tags = _create_tags_from_labeling_job(figure.tags, project_meta)
                label = Label(obj_class=obj_class, geometry=geometry, tags=tags)
                labels.append(label)
            return labels

        if image_ids is None and image_infos is None:
            raise ValueError("Either 'image_ids' or 'image_infos' must be provided.")
        if image_infos is not None:
            image_ids = [image_info.id for image_info in image_infos]
        if self._api.headers.get("x-job-id") != str(id):
            self._api.add_header("x-job-id", str(id))
        job_info = self.get_info_by_id(id)
        if image_infos is None:
            image_infos = self._api.image.get_list(
                job_info.dataset_id,
                filters=[{ApiField.FIELD: ApiField.ID, "operator": "in", "value": image_ids}],
            )
        figures_map = self._api.image.figure.download(job_info.dataset_id, image_ids)
        if self._api.headers.get("x-job-id") == str(id):
            self._api.pop_header("x-job-id")

        if project_meta is None:
            project_meta = self.get_project_meta(id)

        anns = []
        for image in image_infos:
            img_figures = figures_map.get(image.id, [])
            img_tags = _create_tags_from_labeling_job(image.tags, project_meta)
            labels = _create_labels_from_labeling_job(img_figures, project_meta)
            ann = Annotation(img_size=(image.height, image.width), labels=labels, img_tags=img_tags)
            anns.append(ann)
        return anns

    def reject_annotations(self, id: int, mode: Literal["all", "unmarked"] = "all") -> None:
        """
        Reject annotations for all or unmarked entities in the labeling job with given id.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param mode: Reject mode. Can be "all" or "unmarked".
        :type mode: str, optional
        :return: None
        :rtype: :class:`NoneType`
        """

        data = {ApiField.ID: id, ApiField.MODE: mode}
        self._api.post("jobs.reject", data)

    def set_entity_review_status(
        self, id: int, entity_id: int, status: Literal["none", "done", "accepted", "rejected"]
    ) -> None:
        """
        Sets review status for entity with given ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param entity_id: Entity ID
        :type entity_id: int
        :param status: New review status for entity
        :type status: str
        :return: None
        :rtype: :class:`NoneType`
        """
        self._api.post(
            "jobs.entities.update-review-status",
            {ApiField.JOB_ID: id, ApiField.ENTITY_ID: entity_id, ApiField.STATUS: status},
        )

    def clone(
        self,
        id: int,
        new_title: Optional[str] = None,
        reviewer_id: Optional[int] = None,
        assignee_ids: Optional[List[int]] = None,
    ) -> List[LabelingJobInfo]:
        """
        Clone Labeling Job with given ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param new_title: New title for the job
        :type new_title: str, optional
        :param reviewer_id: ID of the reviewer
        :type reviewer_id: int, optional
        :param assignee_ids: List of User IDs to assign the job
        :type assignee_ids: List[int], optional
        :return: List of information about Labeling Jobs. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[LabelingJobInfo]`
        """
        job_info = self.get_info_by_id(id)

        if new_title is None:
            new_title = job_info.name
        if reviewer_id is None:
            reviewer_id = job_info.reviewer_id
        else:
            self._check_membership([reviewer_id], job_info.team_id)
        if assignee_ids is None:
            assignee_ids = [job_info.assigned_to_id]
        else:
            self._check_membership(assignee_ids, job_info.team_id)

        job_infos = self.create(
            name=new_title,
            dataset_id=job_info.dataset_id,
            user_ids=assignee_ids,
            readme=job_info.readme,
            description=job_info.description,
            classes_to_label=job_info.classes_to_label,
            objects_limit_per_image=job_info.objects_limit_per_image,
            tags_to_label=job_info.tags_to_label,
            tags_limit_per_image=job_info.tags_limit_per_image,
            include_images_with_tags=job_info.include_images_with_tags,
            exclude_images_with_tags=job_info.exclude_images_with_tags,
            images_range=job_info.images_range,
            reviewer_id=reviewer_id,
            images_ids=[entity["id"] for entity in job_info.entities],
            # TODO dynamic_classes=job_info.dynamic_classes,
            # TODO dynamic_tags=job_info.dynamic_tags,
        )

        return job_infos

    def restart(
        self,
        id: int,
        assignee_ids: Optional[List[int]] = None,
        reviewer_id: Optional[int] = None,
        title: Optional[str] = None,
        complete_existing: Optional[bool] = True,
        only_rejected_entities: Optional[bool] = True,
        ignore_no_rejected_error: Optional[bool] = False,
    ) -> List[dict]:
        """
        Restart Labeling Job with given ID.

        :param id: Labeling Job ID in Supervisely.
        :type id: int
        :param assignee_ids: List of User IDs to assign the job. If not set, the job will be assigned to the same user as the existing job.
        :type assignee_ids: List[int], optional
        :param reviewer_id: ID of the reviewer
        :type reviewer_id: int, optional
        :param title: New title for the job <= 255 characters
        :type title: str, optional
        :param complete_existing: If False, existing job will not be completed.
        :type complete_existing: bool, optional
        :param only_rejected_entities: If False, all entities that do not have an "accepted" status will be included in new job, all unmarked entities will be rejected for the existing job.
        :type only_rejected_entities: bool, optional
        :param ignore_errors: If True, the job will not be restarted if there are errors in request data.
        :type ignore_errors: bool, optional
        :return: List of dicts with information about created Labeling Jobs.
        :rtype: :class:`List[dict]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api("https://app.supervisely.com", "your_api_token")

            job_info_list = api.labeling_job.restart(222)

            print(job_info_list)
            # Output:
            #   [
            #       {
            #           'id': 940,
            #           'userId': 342,
            #           'type': 'annotation',
            #           'name': 'Annotation Job (#2)'
            #       }
            #   ]
        """

        job_info = self.get_info_by_id(id)

        data = {ApiField.ID: id, ApiField.COMPLETE_EXISTING: complete_existing}

        if assignee_ids is not None:
            self._check_membership(assignee_ids, job_info.team_id)
            data[ApiField.USER_IDS] = assignee_ids
        else:
            data[ApiField.USER_IDS] = [job_info.assigned_to_id]

        if reviewer_id is not None:
            self._check_membership([reviewer_id], job_info.team_id)
            data[ApiField.REVIEWER_ID] = reviewer_id
        else:
            data[ApiField.REVIEWER_ID] = job_info.reviewer_id

        if title is not None:
            data[ApiField.NAME] = title

        if only_rejected_entities is False:
            self.reject_annotations(id, mode="unmarked")

        if only_rejected_entities is True and ignore_no_rejected_error is True:
            try:
                response = self._api.post("jobs.restart", data).json()
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400 and "No images found" in e.response.text:
                    return []
                else:
                    raise e

        response = self._api.post("jobs.restart", data).json()
        return response
