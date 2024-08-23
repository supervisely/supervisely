# coding: utf-8
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

from supervisely.api.module_api import ApiField, ModuleApiBase

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
    parent_id: int
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

        # Get information about the first issue in the list.
        issue_info = api.issues.get_info_by_id(issues[0].id)

        # Add new issue.
        new_issue = api.issues.add(team_id=1, issue_name="New issue", comment="Some comment")
    """

    @staticmethod
    def info_sequence():
        """List of fields that are returned by the API to represent IssueInfo."""
        return [
            ApiField.ID,
            ApiField.STATUS,
            ApiField.PARENT_ID,
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
            "issues.list", {ApiField.FILTER: filters or [], ApiField.TEAM_ID: team_id}
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
        status: Optional[str] = None,
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
            updated_issue = api.issues.update(issue_id=1, issue_name="Updated issue name")"""
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
            "issues.comments.add", {ApiField.ISSUE_ID: issue_id, ApiField.COMMENT: comment}
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
            "issues.comments.editInfo", {ApiField.ID: comment_id, ApiField.COMMENT: comment}
        )

        return CommentInfo.from_json(response.json())
