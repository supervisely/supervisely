# coding: utf-8
"""create or manipulate already existing labeling queues"""

# docs
from __future__ import annotations

from typing import Dict, List, Literal, NamedTuple, Optional, Union

from supervisely.api.constants import PLAYBACK_RATE_POSSIBLE_VALUES
from supervisely.api.labeling_job_api import LabelingJobApi
from supervisely.api.module_api import (
    ApiField,
    ModuleApi,
    ModuleWithStatus,
    RemoveableBulkModuleApi,
)
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.project_type import ProjectType


class LabelingQueueInfo(NamedTuple):
    id: int
    name: str
    team_id: int
    project_id: int
    dataset_id: int
    created_by_id: int
    labelers: list
    reviewers: list
    created_at: str
    finished_at: str
    status: str
    jobs: list
    entities_count: int
    accepted_count: int
    annotated_count: int
    in_progress_count: int
    pending_count: int
    meta: dict


class LabelingQueueApi(RemoveableBulkModuleApi, ModuleWithStatus):
    """
    API for working with Labeling Queues. :class:`LabelingQueueApi<LabelingQueueApi>` object is immutable.

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

        queue = api.labeling_queues.get_info_by_id(2) # api usage example
    """

    @staticmethod
    def info_sequence():
        """
        NamedTuple LabelingQueueInfo information about Labeling Queue.

        :Example:

         .. code-block:: python

            LabelingQueueInfo(
                id=2,
                name='Annotation Queue (#1)',
                team_id=4,
                project_id=58,
                dataset_id=54,
                created_by_id=4,
                labelers=[4],
                reviewers=[4],
                created_at='2020-04-08T15:10:12.618Z',
                finished_at='2020-04-08T15:13:39.788Z',
                status='completed',
                jobs=[283, 282, 281],
                entities_count=3,
                accepted_count=2,
                annotated_count=3,
                in_progress_count=2,
                pending_count=1,
                meta={}
            )
        """
        return [
            ApiField.ID,
            ApiField.NAME,
            ApiField.TEAM_ID,
            ApiField.PROJECT_ID,
            ApiField.DATASET_ID,
            ApiField.CREATED_BY_ID,
            ApiField.LABELERS,
            ApiField.REVIEWERS,
            ApiField.CREATED_AT,
            ApiField.FINISHED_AT,
            ApiField.STATUS,
            ApiField.JOBS,
            ApiField.ENTITIES_COUNT,
            ApiField.ACCEPTED_COUNT,
            ApiField.ANNOTATED_COUNT,
            ApiField.IN_PROGRESS_COUNT,
            ApiField.PENDING_COUNT,
            ApiField.META,
        ]

    @staticmethod
    def info_tuple_name():
        """
        NamedTuple name - **LabelingQueueInfo**.
        """
        return "LabelingQueueInfo"

    def _convert_json_info(self, info: Dict, skip_missing: Optional[bool] = True):
        """ """

        def _get_value(dict, field_name, skip_missing):
            if skip_missing is True:
                return dict.get(field_name, None)
            else:
                return dict[field_name]

        def _get_ids(data, skip_missing):
            if skip_missing is True:
                return [job.get(ApiField.ID, None) for job in data]
            else:
                return [job[ApiField.ID] for job in data]

        if info is None:
            return None
        else:
            field_values = []
            for field_name in self.info_sequence():
                if type(field_name) is str:
                    if field_name in [ApiField.JOBS, ApiField.LABELERS, ApiField.REVIEWERS]:
                        field_values.append(_get_ids(info[field_name], skip_missing))
                        continue
                    else:
                        field_values.append(_get_value(info, field_name, skip_missing))
                elif type(field_name) is tuple:
                    value = None
                    for sub_name in field_name[0]:
                        if value is None:
                            value = _get_value(info, sub_name, skip_missing)
                        else:
                            value = _get_value(value, sub_name, skip_missing)
                    field_values.append(value)
                else:
                    raise RuntimeError("Can not parse field {!r}".format(field_name))
            res = self.InfoType(*field_values)
            return LabelingQueueInfo(**res._asdict())

    def _check_membership(self, ids: List[int], team_id: int) -> None:
        """Check if user is a member of the team in which the Labeling Queue is created."""
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
        user_ids: List[int],
        reviewer_ids: List[int],
        dataset_id: Optional[int] = None,
        collection_id: Optional[int] = None,
        readme: Optional[str] = None,
        classes_to_label: Optional[List[str]] = None,
        objects_limit_per_image: Optional[int] = None,
        tags_to_label: Optional[List[str]] = None,
        tags_limit_per_image: Optional[int] = None,
        include_images_with_tags: Optional[List[str]] = None,
        exclude_images_with_tags: Optional[List[str]] = None,
        images_range: Optional[List[int, int]] = None,
        reviewer_id: Optional[int] = None,
        images_ids: Optional[List[int]] = None,
        dynamic_classes: Optional[bool] = False,
        dynamic_tags: Optional[bool] = False,
        disable_confirm: Optional[bool] = None,
        disable_submit: Optional[bool] = None,
        toolbox_settings: Optional[Dict] = None,
        hide_figure_author: Optional[bool] = False,
        allow_review_own_annotations: Optional[bool] = False,
        skip_complete_job_on_empty: Optional[bool] = False,
        enable_quality_check: Optional[bool] = None,
        quality_check_user_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Creates Labeling Queue and assigns given Users to it.

        :param name: Labeling Queue name in Supervisely.
        :type name: str
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int
        :param collection_id: Entities Collection ID in Supervisely.
        :type collection_id: int, optional
        :param user_ids: User IDs in Supervisely to assign Users as labelers to Labeling Queue.
        :type user_ids: List[int]
        :param reviewer_ids: User IDs in Supervisely to assign Users as reviewers to Labeling Queue.
        :type reviewer_ids: List[int]
        :param readme: Additional information about Labeling Queue.
        :type readme: str, optional
        :param description: Description of Labeling Queue.
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
        :param reviewer_id: User ID in Supervisely to assign User as Reviewer to Labeling Queue.
        :type reviewer_id: int, optional
        :param images_ids: List of images ids to label in dataset
        :type images_ids: List[int], optional
        :param dynamic_classes: If True, classes created after creating the queue will be available for annotators
        :type dynamic_classes: bool, optional
        :param dynamic_tags: If True, tags created after creating the queue will be available for annotators
        :type dynamic_tags: bool, optional
        :param disable_confirm: If True, the Confirm button will be disabled in the labeling tool. It will remain disabled until the next API call sets the parameter to False, re-enabling the button.
        :type disable_confirm: bool, optional
        :param disable_submit: If True, the Submit button will be disabled in the labeling tool. It will remain disabled until the next API call sets the parameter to False, re-enabling the button.
        :type disable_submit: bool, optional
        :param toolbox_settings: Settings for the labeling tool. Only video projects are supported.
        :type toolbox_settings: Dict, optional
        :param hide_figure_author: If True, hides the author of the figure in the labeling tool.
        :type hide_figure_author: bool, optional
        :param allow_review_own_annotations: If True, allows labelers to review their own annotations.
        :type allow_review_own_annotations: bool, optional
        :param skip_complete_job_on_empty: If True, skips completing the Labeling Queue if there are no images to label.
        :type skip_complete_job_on_empty: bool, optional
        :param enable_quality_check: If True, adds an intermediate step between "review" and completing the Labeling Queue.
        :type enable_quality_check: bool, optional
        :param quality_check_user_ids: List of User IDs in Supervisely to assign Users as Quality Checkers to Labeling Queue.
        :type quality_check_user_ids: List[int], optional
        :return: Labeling Queue ID in Supervisely.
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            user_name = 'alex'
            dataset_id = 602
            new_labeling_queue_id = api.labeling_queue.create(
                user_name,
                dataset_id,
                user_ids=[111, 222],
                readme='Readmy text',
                description='Work for labelers',
                objects_limit_per_image=5,
                tags_limit_per_image=3
            )
            print(new_labeling_queue_id)

            # >>> 2

            # Create video labeling job with toolbox settings

            user_id = 4
            dataset_id = 277
            video_id = 24897
            toolbox_settings = {"playbackRate": 32, "skipFramesSize": 15, "showVideoTime": True}

            new_labeling_queue_id = api.labeling_queue.create(
                name="Labeling Queue name",
                dataset_id=dataset_id,
                user_ids=[user_id],
                readme="Labeling Queue readme",
                description="Some description",
                classes_to_label=["car", "animal"],
                tags_to_label=["animal_age_group"],
                images_ids=[video_id],
                toolbox_settings=toolbox_settings,
            )
            print(new_labeling_queue_id)

            # >>> 3
        """
        if dataset_id is None and collection_id is None:
            raise RuntimeError("Either dataset_id or collection_id must be provided")
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
            "dynamicClasses": dynamic_classes,
            "dynamicTags": dynamic_tags,
            "hideFigureAuthor": hide_figure_author,
        }
        if images_ids is not None:
            meta["entityIds"] = images_ids

        if toolbox_settings is not None:
            if dataset_id is not None:
                dataset_info = self._api.dataset.get_info_by_id(dataset_id)
                project_id = dataset_info.project_id
            else:
                collection_info = self._api.entities_collection.get_info_by_id(collection_id)
                project_id = collection_info.project_id
            project_info = self._api.project.get_info_by_id(project_id)
            project_type = project_info.type
            if project_type == ProjectType.VIDEOS.value:
                playback_rate = toolbox_settings.get("playbackRate", None)
                if playback_rate is not None:
                    if playback_rate not in PLAYBACK_RATE_POSSIBLE_VALUES:
                        raise ValueError(
                            f"'playbackRate' must be one of: '{','.join(PLAYBACK_RATE_POSSIBLE_VALUES)}'"
                        )
                meta["toolboxSettings"] = toolbox_settings

        if disable_confirm is not None:
            meta.update({"disableConfirm": disable_confirm})
        if disable_submit is not None:
            meta.update({"disableSubmit": disable_submit})
        if enable_quality_check is not None:
            if quality_check_user_ids is None:
                raise RuntimeError(
                    "quality_check_user_ids must be provided if enable_quality_check is True"
                )
            meta.update({"enableIntermediateReview": enable_quality_check})

        queue_meta = {}
        if allow_review_own_annotations is True:
            queue_meta["reviewOwnAnnotationsAvailable"] = True

        if skip_complete_job_on_empty is True:
            queue_meta["skipCompleteAnnotationJobOnEmpty"] = True

        data = {
            ApiField.NAME: name,
            ApiField.USER_IDS: user_ids,
            ApiField.REVIEWER_IDS: reviewer_ids,
            ApiField.META: meta,
        }
        if dataset_id is not None:
            data[ApiField.DATASET_ID] = dataset_id
        if collection_id is not None:
            data[ApiField.COLLECTION_ID] = collection_id
        if quality_check_user_ids is not None:
            if enable_quality_check is not True:
                raise RuntimeError(
                    "quality_check_user_ids can be set only if enable_quality_check is True"
                )
            data[ApiField.QUALITY_CHECK_USER_IDS] = quality_check_user_ids

        if len(queue_meta) > 0:
            data[ApiField.QUEUE_META] = queue_meta

        if readme is not None:
            data[ApiField.README] = str(readme)

        if images_range is not None and images_range != (None, None):
            if len(images_range) != 2:
                raise RuntimeError("images_range has to contain 2 elements (start, end)")
            images_range = {"start": images_range[0], "end": images_range[1]}
            data[ApiField.META]["range"] = images_range

        if reviewer_id is not None:
            data[ApiField.REVIEWER_ID] = reviewer_id

        response = self._api.post("labeling-queues.add", data)
        return response.json()["id"]  # {"success": true}

    def get_list(
        self,
        team_id: int,
        dataset_id: Optional[int] = None,
        project_id: Optional[int] = None,
        ids: Optional[List[int]] = None,
        names: Optional[List[str]] = None,
        show_disabled: Optional[bool] = False,
    ) -> List[LabelingQueueInfo]:
        """
        Get list of information about Labeling Queues in the given Team.

        :param team_id: Team ID in Supervisely.
        :type team_id: int
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int, optional
        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :param ids: List of Labeling Queue IDs in Supervisely.
        :type ids: List[int], optional
        :param names: List of Labeling Queue names in Supervisely.
        :type names: List[str], optional
        :param show_disabled: Show disabled Labeling Queues.
        :type show_disabled: bool, optional
        :return: List of information about Labeling Queues. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`List[LabelingQueueInfo]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            label_jobs = api.labeling_queue.get_list(4)
        """

        filters = []
        if project_id is not None:
            filters.append({"field": ApiField.PROJECT_ID, "operator": "=", "value": project_id})
        if dataset_id is not None:
            filters.append({"field": ApiField.DATASET_ID, "operator": "=", "value": dataset_id})
        if names is not None:
            filters.append({"field": ApiField.NAME, "operator": "in", "value": names})
        if ids is not None:
            filters.append({"field": ApiField.ID, "operator": "in", "value": ids})
        return self.get_list_all_pages(
            "labeling-queues.list",
            {ApiField.TEAM_ID: team_id, "showDisabled": show_disabled, ApiField.FILTER: filters},
        )

    def get_info_by_id(self, id: int) -> LabelingQueueInfo:
        """
        Get Labeling Queue information by ID.

        :param id: Labeling Queue ID in Supervisely.
        :type id: int
        :return: Information about Labeling Queue. See :class:`info_sequence<info_sequence>`
        :rtype: :class:`LabelingJobInfo`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            label_job_info = api.labeling_queue.get_info_by_id(2)
            print(label_job_info)
            # Output: [
            #     2,
            #     "Annotation Queue (#1) (#1) (dataset_01)",
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
        return self._get_info_by_id(id, "labeling-queues.info")

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

            job_status = api.labeling_queue.get_status(4)
            print(job_status) # pending
        """
        status_str = self.get_info_by_id(id).status
        return LabelingJobApi.Status(status_str)

    def set_status(self, id: int, status: str) -> None:
        """
        Sets Labeling Queue status.

        :param id: Labeling Queue ID in Supervisely.
        :type id: int
        :param status: New Labeling Queue status
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

            api.labeling_queue.set_status(id=9, status="completed")
        """
        self._api.post("labeling-queues.set-status", {ApiField.ID: id, ApiField.STATUS: status})

    def get_project_meta(self, id: int) -> ProjectMeta:
        """
        Returns project meta with classes and tags used in the labeling queue with given id.

        :param id: Labeling Queue ID in Supervisely.
        :type id: int
        :return: Project meta of the labeling queue with given id.
        :rtype: :class:`ProjectMeta`
        """
        queue_info = self.get_info_by_id(id)
        project_meta_json = self._api.project.get_meta(queue_info.project_id, with_settings=True)
        project_meta = ProjectMeta.from_json(project_meta_json)

        jobs = [self._api.labeling_job.get_info_by_id(job_id) for job_id in queue_info.jobs]
        job_classes = set()
        job_tags = set()
        for job in jobs:
            job_classes.update(job.classes_to_label)
            job_tags.update(job.tags_to_label)

        filtered_classes = [
            obj_cls for obj_cls in project_meta.obj_classes if obj_cls.name in job_classes
        ]
        filtered_tags = [
            tag_meta for tag_meta in project_meta.tag_metas if tag_meta.name in job_tags
        ]
        queue_meta = ProjectMeta(obj_classes=filtered_classes, tag_metas=filtered_tags)
        return queue_meta

    def get_entities_all_pages(
        self,
        id: int,
        collection_id: Optional[int] = None,
        per_page: int = 25,
        sort: str = "name",
        sort_order: str = "asc",
        status: Optional[Union[List, Literal["none", "done", "accepted", "null"]]] = None,
        limit: int = None,
        filter_by: List[Dict] = None,
    ) -> Dict[str, Union[List[Dict], int]]:
        """
        Get list of all or limited quantity entities from the Supervisely server.

        :param id: Labeling Queue ID in Supervisely.
        :type id: int
        :param collection_id: Collection ID in Supervisely.
        :type collection_id: int, optional
        :param per_page: Number of entities per page.
        :type per_page: int, optional
        :param sort: Sorting field.
        :type sort: str, optional
        :param sort_order: Sorting order.
        :type sort_order: str, optional
        :param status: Status of entities to filter.
                        "null" - pending (in queue).
                        "none" - annotating (not in queue),
                        "done" - on review,
                        "accepted" - accepted,
        :type status: str or List[str], optional
        :param limit: Limit the number of entities to return. If limit is None, all entities will be returned.
        :type limit: int, optional
        :param filter_by: Filter for entities.
                       e.g. [{"field": "name", "operator": "in", "value": ["image_01", "image_02"]}]
                        - field - field name to filter by ("id", "name", "reviewedAt")
                        - operator - operator to use for filtering ("=", ">", "<", ">=", "<=")
                        - value - value to filter by
        :type filter_by: List[Dict], optional
        :param return_first_response: Specify if return first response
        :type return_first_response: bool, optional
        """

        method = "labeling-queues.stats.entities"
        data = {
            ApiField.ID: id,
            ApiField.PAGE: 1,
            ApiField.PER_PAGE: per_page,
            ApiField.SORT: sort,
            ApiField.SORT_ORDER: sort_order,
        }
        if collection_id is not None:
            data[ApiField.FILTERS] = [
                {
                    "type": "entities_collection",
                    "data": {ApiField.COLLECTION_ID: collection_id, ApiField.INCLUDE: True},
                }
            ]
        if filter_by is not None:
            data[ApiField.FILTER] = filter_by
        if status is not None:
            if type(status) is str:
                status = [status]
            status = [None if s == "null" else s.lower() for s in status]
            data["entityStatus"] = status

        first_response = self._api.post(method, data).json()
        total = first_response["total"]
        pages_count = int(total / per_page) + 1 if total % per_page != 0 else int(total / per_page)
        if pages_count in [0, 1]:
            return first_response

        limit_exceeded = False
        results = first_response["images"]
        if limit is not None and len(results) > limit:
            limit_exceeded = True

        if (pages_count == 1 and len(results) == total) or limit_exceeded is True:
            pass
        else:
            for page_idx in range(2, pages_count + 1):
                temp_resp = self._api.post(method, {**data, "page": page_idx, "per_page": per_page})
                temp_items = temp_resp.json()["images"]
                results.extend(temp_items)
                if limit is not None and len(results) > limit:
                    limit_exceeded = True
                    break

            if len(results) != total and limit is None:
                raise RuntimeError(
                    "Method {!r}: error during pagination, some items are missed".format(method)
                )

        if limit is not None:
            results = results[:limit]
        return {"images": results, "total": total}

    def get_entities_count_by_status(
        self,
        id: int,
        status: Optional[Union[List, Literal["none", "done", "accepted", "null"]]] = None,
        filter_by: List[Dict] = None,
    ) -> int:
        """
        Get count of entities in the given Labeling Queue with given status.
        :param id: Labeling Queue ID in Supervisely.
        :type id: int
        :param status: Status of entities to filter.
                        "null" - pending (in queue).
                        "none" - annotating (not in queue),
                        "done" - on review,
                        "accepted" - accepted,
        :type status: str or List[str], optional
        :param filter_by: Filter for entities.
                       e.g. [{"field": "name", "operator": "in", "value": ["image_01", "image_02"]}]
                        - field - field name to filter by ("id", "name", "reviewedAt")
                        - operator - operator to use for filtering ("=", ">", "<", ">=", "<=")
                        - value - value to filter by
        :type filter_by: List[Dict], optional
        :return: Count of entities in the Labeling Queue with given status.
        :rtype: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            os.environ['SERVER_ADDRESS'] = 'https://app.supervisely.com'
            os.environ['API_TOKEN'] = 'Your Supervisely API Token'
            api = sly.Api.from_env()

            entities_count = api.labeling_queue.get_entities_count_by_status(4, status="none")
            print(entities_count)
            # Output: 3
        """
        return self.get_entities_all_pages(id, status=status, limit=1, filter_by=filter_by).get(
            "total", 0
        )
