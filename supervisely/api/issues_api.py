# coding: utf-8
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
        """Create an instance of the class from JSON data.

        :param data: JSON data.
        :type data: Dict
        :return: Instance of the class.
        :rtype: CommentInfo"""
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


class IssuesApi(ModuleApiBase):
    """Class for working with issues in Supervisely.

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

        # Get list of issues in specified team.
        issues = api.issues.get_list(team_id=1)

        # Get information about issue by its ID.
        issue_info = api.issues.get_info_by_id(id=1)

        # Add new issue.
        new_issue = api.issues.add(team_id=1, issue_name="New issue", comment="Some comment")
    """

    @staticmethod
    def info_sequence():
        """List of fields that are returned by the API to represent IssueInfo."""
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
        ]

    @staticmethod
    def info_tuple_name():
        """Name of the tuple that represents IssueInfo."""
        return "IssueInfo"

    def get_list(self, team_id: int, filters: List[Dict[str, str]] = None) -> List[IssueInfo]:
        """Get list of issues in the specified team.

        :param team_id: Team ID.
        :type team_id: int
        :param filters: List of filters to apply to the list of issues.
        :type filters: List[Dict[str, str]], optional

        :return: List of issues.
        :rtype: List[IssueInfo]

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

            # Get list of issues in specified team.
            issues = api.issues.get_list(team_id=1)
        """
        return self.get_list_all_pages(
            "issues.list",
            {ApiField.FILTER: filters or [], ApiField.TEAM_ID: team_id},
        )

    def get_info_by_id(self, id: int) -> IssueInfo:
        """Get information about the issue by its ID.

        :param id: Issue ID.
        :type id: int
        :return: Information about the issue.
        :rtype: IssueInfo

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

            # Get information about the issue by its ID.
            issue_info = api.issues.get_info_by_id(1)"""
        response = self._get_response_by_id(id, "issues.info", id_field=ApiField.ID)
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
        :return: Information about the added issue.
        :rtype: IssueInfo

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
        :raises ValueError: If the status is incorrect.
        :return: Information about the issue.
        :rtype: IssueInfo

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

            # Update information about the issue.
            updated_issue = api.issues.update(issue_id=1, issue_name="Updated issue name")
        """
        available_statuses = ["open", "closed"]
        if status is not None and status not in available_statuses:
            raise ValueError(
                f"Incorrect status, expected one of {available_statuses}, got {status}"
            )
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
        """Remove the issue by its ID.
        NOTE: This operation is irreversible.

        :param issue_id: Issue ID.
        :type issue_id: int

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

            # Remove the issue by its ID.
            api.issues.remove(issue_id=1)"""
        self._api.post("issues.remove", {ApiField.ID: issue_id})

    def add_comment(self, issue_id: int, comment: str) -> CommentInfo:
        """Add a comment to the issue with the specified ID.

        :param issue_id: Issue ID.
        :type issue_id: int
        :param comment: Comment text.
        :type comment: str

        :return: Information about the added comment.
        :rtype: CommentInfo

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

            # Add a comment to the issue with the specified ID.
            comment_info = api.issues.add_comment(issue_id=1, comment="Some comment")"""
        response = self._api.post(
            "issues.comments.add",
            {ApiField.ISSUE_ID: issue_id, ApiField.COMMENT: comment},
        )

        return CommentInfo.from_json(response.json())

    def update_comment(self, comment_id: int, comment: str) -> CommentInfo:
        """Update the comment with the specified ID.

        :param comment_id: Comment ID.
        :type comment_id: int
        :param comment: New comment text.
        :type comment: str
        :return: Information about the updated comment.
        :rtype: CommentInfo

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

            # Update the comment with the specified ID.
            api.issues.update_comment(comment_id=1, comment="Updated comment")"""
        response = self._api.post(
            "issues.comments.editInfo",
            {ApiField.ID: comment_id, ApiField.COMMENT: comment},
        )

        return CommentInfo.from_json(response.json())

    def _create_bindings(
        self, label_id: int, image_id: int
    ) -> Dict[str, Union[str, int, Dict[str, int]]]:
        """Create bindings from the given parameters.

        :param label_id: Label ID.
        :type label_id: int
        :param image_id: Image ID.
        :type image_id: int
        :return: Bindings.
        :rtype: Dict[str, Union[str, int, Dict[str, int]]]"""
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
        """Add a subissue to the specified issue.
        Image and label IDs should be the same type, e.g. both int or list of ints.
        If they are lists, they should have the same length.
        Annotation info should be an instance of AnnotationInfo, not sly.Annotation, since the
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
        :type annotation_info: AnnotationInfo
        :param project_meta: Project meta information.
        :type project_meta: ProjectMeta

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
