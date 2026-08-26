# coding: utf-8
"""Create and manage annotation issues and comments in Supervisely."""

from __future__ import annotations

from typing import Dict, List, Literal, NamedTuple, Optional, Union

from supervisely.api.annotation_api import AnnotationInfo
from supervisely.api.module_api import ApiField, ModuleApiBase
from supervisely.project.project_meta import ProjectMeta

# TODO: Update autodocs configuration to include this module.


class CommentInfo(NamedTuple):
    """Class that represents information about a comment."""

    id: int
    issue_id: int
    created_by: int
    created_at: str
    comment: str
    meta: Dict
    created_by_user: str
    links: Dict

    @classmethod
    def from_json(cls, data: Dict) -> CommentInfo:
        """
        Create an instance of the class from JSON data.

        :param data: JSON data.
        :type data: Dict
        :returns: Instance of the class.
        :rtype: :class:`~supervisely.api.issues_api.CommentInfo`
        """
        return cls(
            id=data.get(ApiField.ID),
            issue_id=data.get(ApiField.ISSUE_ID),
            created_by=data.get(ApiField.CREATED_BY_ID[0][0]),
            created_at=data.get(ApiField.CREATED_AT),
            comment=data.get(ApiField.COMMENT),
            meta=data.get(ApiField.META),
            created_by_user=data.get(ApiField.CREATED_BY_USER),
            links=data.get(ApiField.LINKS),
        )


class IssueInfo(NamedTuple):
    """Class that represents information about an issue."""

    id: int
    status: str
    user_login: str
    image_id: int
    created_by: int
    created_at: str
    updated_at: str
    dataset_id: int
    project_id: int
    image_name: str
    name: str
    sub_issues: Optional[List[Dict]]


class IssuesApi(ModuleApiBase):
    """API for working with annotation issues and comments."""

    def __init__(self, api):
        """
        :param api: :class:`~supervisely.api.api.Api` object to use for API connection.
        :type api: :class:`~supervisely.api.api.Api`

        :Usage Example:

            .. code-block:: python

                import supervisely as sly
                api = sly.Api.from_env()
                issues = api.issues.get_list(team_id=1)
        """
        super().__init__(api)

    _AVAILABLE_STATUSES = ["open", "closed"]

    @classmethod
    def _validate_status(cls, status: Optional[str]) -> None:
        """
        Check that ``status`` (if given) is one of the statuses the API accepts.

        :param status: Status to validate.
        :type status: str, optional
        :raises ValueError: if the status is incorrect. Expected one of ["open", "closed"], got {status}
        :returns: None
        :rtype: None
        """
        if status is not None and status not in cls._AVAILABLE_STATUSES:
            raise ValueError(
                f"Incorrect status, expected one of {cls._AVAILABLE_STATUSES}, got {status}"
            )

    def _validate_project_and_dataset_id(
        self, project_id: Optional[int], dataset_id: Optional[int]
    ) -> None:
        """
        Check if only one of 'project_id' and 'dataset_id' is provided.

        :param project_id: Project ID in Supervisely.
        :type project_id: int, optional
        :param dataset_id: Dataset ID in Supervisely.
        :type dataset_id: int, optional
        :raises ValueError: if both 'project_id' and 'dataset_id' are provided or none of them are provided.
        :returns: None
        :rtype: None
        """
        if project_id is None and dataset_id is None:
            raise ValueError("One of 'project_id' or 'dataset_id' should be provided.")

        if project_id is not None and dataset_id is not None:
            raise ValueError("Only one of 'project_id' and 'dataset_id' should be provided.")

    @staticmethod
    def info_sequence():
        """Sequence of fields that are returned by the API to represent IssueInfo."""
        return [
            ApiField.ID,
            ApiField.STATUS,
            ApiField.USER_LOGIN,
            ApiField.IMAGE_ID,
            ApiField.CREATED_BY_ID,
            ApiField.CREATED_AT,
            ApiField.UPDATED_AT,
            ApiField.DATASET_ID,
            ApiField.PROJECT_ID,
            ApiField.IMAGE_NAME,
            ApiField.NAME,
            ApiField.SUB_ISSUES,
        ]

    @staticmethod
    def info_tuple_name():
        """Name of the tuple that represents IssueInfo."""
        return "IssueInfo"

    def get_list(
        self,
        team_id: int,
        filters: List[Dict[str, str]] = None,
        with_sub_issues: bool = False,
    ) -> List[IssueInfo]:
        """Get list of issues in the specified team.

        NOTE on ``with_sub_issues``: has the same partial-resolution behavior as
        :meth:`get_info_by_id`'s ``with_sub_issues`` — ``imageId`` is only resolved for a
        sub-issue bound to a specific labeled object (``figureId``); a sub-issue bound directly
        to a whole image (``imageId``) still comes back with ``imageId: None``. Use
        :meth:`get_list_by_dataset` if you need that case resolved too.

        :param team_id: Team ID.
        :type team_id: int
        :param filters: List of filters to apply to the list of issues.
        :type filters: List[Dict[str, str]], optional
        :param with_sub_issues: Whether to include each issue's sub-issues in the response.
        :type with_sub_issues: bool, optional
        :returns: List of issues.
        :rtype: List[:class:`~supervisely.api.issues_api.IssueInfo`]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Get list of issues in specified team.
                issues = api.issues.get_list(team_id=1)

                # Get list of issues together with their sub-issues.
                issues = api.issues.get_list(team_id=1, with_sub_issues=True)
        """
        def _convert(item):
            # issues.list omits subIssues from its response when with_sub_issues=False —
            # default it explicitly rather than relaxing skip_missing for every field.
            item.setdefault(ApiField.SUB_ISSUES, None)
            return self._convert_json_info(item)

        return self.get_list_all_pages(
            "issues.list",
            {
                ApiField.FILTER: filters or [],
                ApiField.TEAM_ID: team_id,
                ApiField.WITH_SUB_ISSUES: with_sub_issues,
            },
            convert_json_info_cb=_convert,
        )

    def get_info_by_id(self, id: int, with_sub_issues: bool = False) -> IssueInfo:
        """Get information about the issue by its ID.

        NOTE: ``with_sub_issues=True`` only resolves the image for a sub-issue that was raised
        against a specific labeled object (bound to a ``figureId``) — that sub-issue's
        ``imageId`` will be populated. A sub-issue raised against a whole image (bound directly
        to an ``imageId``) is **not** resolved by this method: it comes back with ``imageId``
        set to ``None`` and no way to recover the image from here. To reliably resolve the
        image for both cases in bulk, use :meth:`get_list_by_dataset` instead.

        :param id: Issue ID.
        :type id: int
        :param with_sub_issues: Whether to include the issue's sub-issues in the response.
        :type with_sub_issues: bool, optional
        :returns: Information about the issue.
        :rtype: :class:`~supervisely.api.issues_api.IssueInfo`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Get information about the issue by its ID.
                issue_info = api.issues.get_info_by_id(1)

                # Get information about the issue together with its sub-issues.
                issue_info = api.issues.get_info_by_id(1, with_sub_issues=True)
        """
        response = self._get_response_by_id(
            id,
            "issues.info",
            id_field=ApiField.ID,
            fields={ApiField.WITH_SUB_ISSUES: with_sub_issues},
        )
        return (
            self._convert_json_info(response.json(), skip_missing=True)
            if (response is not None)
            else None
        )

    def add(
        self,
        team_id: int,
        issue_name: str,
        comment: Optional[str] = None,
        assignees: Optional[List[int]] = None,
        is_local: bool = False,
    ) -> IssueInfo:
        """Add a new issue and return information about it.

        :param team_id: Team ID.
        :type team_id: int
        :param issue_name: Name of the issue.
        :type issue_name: str
        :param comment: Comment for the issue.
        :type comment: str, optional
        :param assignees: List of user IDs to assign the issue.
        :type assignees: List[int], optional
        :param is_local: The local issue will be available only for the members of the team, where it was
            created. If set to False, the issue will be available for all users from all teams.
        :type is_local: bool
        :returns: Information about the added issue.
        :rtype: :class:`~supervisely.api.issues_api.IssueInfo`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Add new issue.
                new_issue = api.issues.add(team_id=1, issue_name="New issue", comment="Some comment")
        """
        response = self._api.post(
            "issues.add",
            {
                ApiField.NAME: issue_name,
                ApiField.COMMENT: comment or "",
                ApiField.ASSIGNEES: assignees or [],
                ApiField.TEAM_ID: team_id,
                ApiField.IS_LOCAL: is_local,
            },
        )
        issue_id = response.json().get(ApiField.ID)
        # * At the moment API returns only ID of the issue (e.g. {"id": 123}).
        # * So, we're making extra request to get full info about the issue.
        # * Consider to update API to return full info about the issue.
        return self.get_info_by_id(issue_id)

    def update(
        self,
        issue_id: int,
        issue_name: Optional[str] = None,
        status: Optional[Literal["open", "closed"]] = None,
        is_pinned: Optional[bool] = None,
    ) -> IssueInfo:
        """Update information about the issue.

        :param issue_id: Issue ID.
        :type issue_id: int
        :param issue_name: New name of the issue.
        :type issue_name: str, optional
        :param status: New status of the issue.
        :type status: str, optional
        :param is_pinned: Whether the issue is pinned.
        :type is_pinned: bool, optional
        :raises ValueError: if the status is incorrect. Expected one of ["open", "closed"], got {status}
        :returns: Information about the issue.
        :rtype: :class:`~supervisely.api.issues_api.IssueInfo`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Update information about the issue.
                updated_issue = api.issues.update(issue_id=1, issue_name="Updated issue name")
        """
        self._validate_status(status)
        payload = {
            ApiField.ID: issue_id,
            ApiField.NAME: issue_name,
            ApiField.STATUS: status,
            ApiField.IS_PINNED: is_pinned,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        self._api.post("issues.editInfo", payload)

        # * Consider to update API to return full info about the issue without extra request.
        return self.get_info_by_id(issue_id)

    def remove(self, issue_id: int) -> None:
        """
        Remove the issue by its ID.
        NOTE: This operation is irreversible.

        :param issue_id: Issue ID.
        :type issue_id: int

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Remove the issue by its ID.
                api.issues.remove(issue_id=1)
        """
        self._api.post("issues.remove", {ApiField.ID: issue_id})

    def add_comment(self, issue_id: int, comment: str) -> CommentInfo:
        """
        Add a comment to the issue with the specified ID.

        :param issue_id: Issue ID.
        :type issue_id: int
        :param comment: Comment text.
        :type comment: str
        :returns: Information about the added comment.
        :rtype: :class:`~supervisely.api.issues_api.CommentInfo`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Add a comment to the issue with the specified ID.
                comment_info = api.issues.add_comment(issue_id=1, comment="Some comment")
        """
        response = self._api.post(
            "issues.comments.add",
            {ApiField.ISSUE_ID: issue_id, ApiField.COMMENT: comment},
        )
        return CommentInfo.from_json(response.json())

    def update_comment(self, comment_id: int, comment: str) -> CommentInfo:
        """
        Update the comment with the specified ID.

        :param comment_id: Comment ID.
        :type comment_id: int
        :param comment: New comment text.
        :type comment: str
        :returns: Information about the updated comment.
        :rtype: :class:`~supervisely.api.issues_api.CommentInfo`

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Update the comment with the specified ID.
                api.issues.update_comment(comment_id=1, comment="Updated comment")
        """
        response = self._api.post(
            "issues.comments.editInfo",
            {ApiField.ID: comment_id, ApiField.COMMENT: comment},
        )
        return CommentInfo.from_json(response.json())

    def _create_bindings(self, label_id: int, image_id: int) -> Dict[str, Union[str, int, Dict[str, int]]]:
        """Create bindings from the given parameters.

        :param label_id: Label ID.
        :type label_id: int
        :param image_id: Image ID.
        :type image_id: int
        :returns: Bindings.
        :rtype: Dict[str, Union[str, int, Dict[str, int]]]
        """
        # NOTE: This method is designed to handle the bindings for different cases,
        # e.g. linking dataset, project, etc. At the moment, it's used for linking
        # the issue with the image. Later, it can be extended to handle other cases.
        # In this case parameters should be optional.
        return {
            ApiField.FIELD: ApiField.FIGURE_ID,
            ApiField.VALUE: label_id,
            ApiField.EXTRA: {ApiField.FIGURE_IMAGE_ID: image_id},
        }

    def add_subissue(
        self,
        issue_id: int,
        image_ids: Union[int, List[int]],
        label_ids: Union[int, List[int]],
        top: Union[int, float],
        left: Union[int, float],
        annotation_info: AnnotationInfo,
        project_meta: ProjectMeta,
    ) -> None:
        """
        Add a subissue to the specified issue.
        Image and label IDs should be the same type, e.g. both int or list of ints.
        If they are lists, they should have the same length.
        Annotation info should be an instance of AnnotationInfo, not :class:`~supervisely.annotation.annotation.Annotation`, since the
        second one does not contain required information.

        :param issue_id: Issue ID.
        :type issue_id: int
        :param image_ids: Image ID or list of image IDs to be binded with the issue.
        :type image_ids: Union[int, List[int]]
        :param label_ids: Label ID or list of label IDs to be binded with the issue.
        :type label_ids: Union[int, List[int]]
        :param top: Top position of the marker of subissue in the Labeling interface.
        :type top: Union[int, float]
        :param left: Left position of the marker of subissue in the Labeling interface.
        :type left: Union[int, float]
        :param annotation_info: Information about the annotation.
        :type annotation_info: :class:`~supervisely.api.annotation_api.AnnotationInfo`
        :param project_meta: Project meta information.
        :type project_meta: :class:`~supervisely.project.project_meta.ProjectMeta`
        :returns: None
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                project_id = 123
                image_id = 456
                label_id = 789

                # Get project meta and annotation info.
                project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
                annotation_info = api.annotation.download(image_id)

                # Add a subissue to the specified issue.
                api.issues.add_subissue(
                    issue_id=1,
                    image_ids=image_id,
                    label_ids=label_id,
                    top=100,
                    left=100,
                    annotation_info=annotation_info,
                    project_meta=project_meta
                )
        """
        # NOTE: DO NOT USE THIS METHOD IN PRODUCTION CODE.
        # From the API side, there will be significant changes in the future which lead to
        # changes in the method signature.
        bindings = self._create_bindings(label_ids, image_ids)
        if type(image_ids) != type(label_ids):
            raise ValueError(
                "Image ID and Label ID should be the same type, e.g. both int or list of ints."
            )
        if isinstance(image_ids, int):
            image_ids = [image_ids]
            label_ids = [label_ids]

        if len(image_ids) != len(label_ids):
            raise ValueError(
                "Image ID and Label ID should have the same length when they are lists."
            )

        if not isinstance(annotation_info, AnnotationInfo):
            raise ValueError("annotation_info should be an instance of AnnotationInfo.")

        bindings = [
            self._create_bindings(label_id, image_id)
            for label_id, image_id in zip(label_ids, image_ids)
        ]

        classes = project_meta.to_json()["classes"]

        annotation_data = annotation_info.to_json()
        annotation_data[ApiField.META] = {ApiField.CLASSES: classes}

        payload = {
            ApiField.ISSUE_ID: issue_id,
            ApiField.BINDINGS: bindings,
            ApiField.META: {
                ApiField.POSITION: {ApiField.LEFT: left, ApiField.TOP: top},
                ApiField.ANNOTATION_DATA: annotation_data,
            },
            ApiField.PARENT_ID: issue_id,
        }

        self._api.post("issues.sub-issue.add", payload)

    def update_subissue(
        self,
        sub_issue_id: int,
        status: Optional[Literal["open", "closed"]] = None,
        parent_id: Optional[int] = None,
        meta: Optional[Dict] = None,
    ) -> None:
        """
        Update a sub-issue, e.g. to close it or move it under a different parent issue.

        NOTE: unlike :meth:`update` (for top-level issues), this returns ``None`` rather than
        the updated record — there is no ``issues.sub-issue.info`` endpoint to re-fetch a
        single sub-issue by id, so this mirrors :meth:`add_subissue`, which has the same
        limitation.

        :param sub_issue_id: Sub-issue ID.
        :type sub_issue_id: int
        :param status: New status of the sub-issue.
        :type status: str, optional
        :param parent_id: ID of the issue to move the sub-issue under.
        :type parent_id: int, optional
        :param meta: New meta information for the sub-issue (e.g. marker ``position``).
        :type meta: dict, optional
        :raises ValueError: if the status is incorrect. Expected one of ["open", "closed"], got {status}
        :returns: None
        :rtype: None

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                # Close a sub-issue.
                api.issues.update_subissue(sub_issue_id=1, status="closed")
        """
        self._validate_status(status)
        payload = {
            ApiField.ID: sub_issue_id,
            ApiField.STATUS: status,
            ApiField.PARENT_ID: parent_id,
            ApiField.META: meta,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        self._api.post("issues.sub-issue.editInfo", payload)

    @staticmethod
    def resolve_binding_image_id(record: Dict) -> Optional[int]:
        """
        Resolve the image ID a :meth:`get_list_by_dataset` record is bound to, if any.

        :param record: One record from :meth:`get_list_by_dataset`.
        :type record: dict
        :returns: The image ID, or ``None`` if the record is a top-level issue container (no
            ``bindings``), or its binding isn't tied to a single image (``projectId``/``jobId``).
        :rtype: int, optional
        """
        bindings = record.get(ApiField.BINDINGS)
        if not bindings:
            return None
        binding = bindings[0]
        field = binding.get(ApiField.FIELD)
        if field == ApiField.IMAGE_ID:
            return binding.get(ApiField.VALUE)
        if field == ApiField.FIGURE_ID:
            return binding.get(ApiField.EXTRA, {}).get(ApiField.FIGURE_IMAGE_ID)
        return None

    def get_list_by_dataset(
        self,
        dataset_id: Optional[int] = None,
        project_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get all issues bound to entities within the specified dataset or project, including
        sub-issues and the raw ``bindings`` that link each sub-issue to its image or labeled
        object.

        Unlike :meth:`get_list`, which only returns top-level issue containers (never linked to
        an image), this method is the reliable way to resolve which image (or object) an issue
        is actually about. Use :meth:`resolve_binding_image_id` to read the image ID off each
        record instead of parsing ``bindings`` by hand.

        NOTE: this endpoint has no per-issue filter and isn't paginated — it always returns
        every issue and sub-issue in the given dataset/project in one response. Resolving a
        single issue's image this way costs a full dataset/project-wide fetch; there's no
        cheaper supported call today for that single-issue case.

        The returned list mixes two kinds of records:

        - Top-level issue containers (no ``bindings`` key) — not bound to any single entity.
        - Sub-issues (have a ``bindings`` key with exactly one entry:
          ``{"field": ..., "value": ..., "extra": {...}}``). ``field`` is one of
          ``"imageId"``, ``"figureId"``, ``"projectId"``, ``"jobId"``. To resolve the image:

          - ``field == "imageId"``: the image ID is ``bindings[0]["value"]``.
          - ``field == "figureId"``: the bound object's (figure's) ID is
            ``bindings[0]["value"]``, and its image ID is
            ``bindings[0]["extra"]["figureImageId"]``.
          - ``field in ("projectId", "jobId")``: not tied to a single image.

        :param dataset_id: Dataset ID. Exactly one of ``dataset_id``/``project_id`` is required.
        :type dataset_id: int, optional
        :param project_id: Project ID. Exactly one of ``dataset_id``/``project_id`` is required.
        :type project_id: int, optional
        :raises ValueError: if neither or both of ``dataset_id``/``project_id`` are given.
        :returns: Raw list of issue and sub-issue records, as described above.
        :rtype: List[dict]

        :Usage Example:

            .. code-block:: python

                import os
                from dotenv import load_dotenv

                import supervisely as sly

                # Load secrets and create API object from .env file (recommended)
                # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
                if sly.is_development():
                    load_dotenv(os.path.expanduser("~/supervisely.env"))

                api = sly.Api.from_env()

                records = api.issues.get_list_by_dataset(dataset_id=123)

                # Count issues per image.
                issues_per_image = {}
                for record in records:
                    image_id = api.issues.resolve_binding_image_id(record)
                    if image_id is not None:
                        issues_per_image[image_id] = issues_per_image.get(image_id, 0) + 1
        """
        self._validate_project_and_dataset_id(project_id, dataset_id)

        payload = {}
        if dataset_id is not None:
            payload[ApiField.DATASET_ID] = dataset_id
        if project_id is not None:
            payload[ApiField.PROJECT_ID] = project_id

        # issues.dataset-issues.list returns a flat, unpaginated array (confirmed against its
        # server-side handler and by live testing) — unlike issues.list, it has no
        # entities/total/perPage envelope, so this deliberately bypasses get_list_all_pages.
        # Same pattern as AdvancedApi.get_object_tags() for another confirmed-unpaginated
        # endpoint (figures.tags.list).

        response = self._api.post("issues.dataset-issues.list", payload)
        return response.json()
