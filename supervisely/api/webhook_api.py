# coding: utf-8
"""API for working with Webhooks"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

from supervisely.api.module_api import ApiField, ModuleNoParent

if TYPE_CHECKING:
    from supervisely.api.api import Api


# Available webhook action types
LABELING_JOB_COMPLETED = "labeling_job.completed"
LABELING_QUEUE_JOB_COMPLETED = "labeling_queue.job.completed"
LABELING_QUEUE_COMPLETED = "labeling_queue.completed"


class WebhookInfo(NamedTuple):
    """Information about a webhook returned from API"""

    id: int
    """Webhook ID"""
    action: str
    """Webhook action type"""
    url: str
    """Target URL"""
    retries_count: int
    """Number of retry attempts"""
    retries_delay: int
    """Delay between retries in seconds"""
    team_id: int
    """Team ID"""
    created_at: str
    """Creation timestamp"""
    updated_at: str
    """Last update timestamp"""


class WebhookApi(ModuleNoParent):
    """
    API for working with Webhooks. :class:`WebhookApi<WebhookApi>` object is immutable.

    :param api: API connection to the server
    :type api: Api
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        if sly.is_development():
            load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Create a webhook
        webhook = api.webhook.create(
            team_id=123,
            url="https://example.com/webhook",
            action=sly.LABELING_JOB_COMPLETED
        )
    """

    def get_list(
        self, team_id: int, filters: Optional[List[Dict]] = None
    ) -> List[WebhookInfo]:
        """
        Get list of all webhooks in a team.

        :param team_id: Team ID
        :type team_id: int
        :param filters: List of params to filter webhooks
        :type filters: List[Dict], optional
        :return: List of webhook information objects
        :rtype: List[WebhookInfo]
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            webhooks = api.webhook.get_list(team_id=123)
        """
        data = {ApiField.TEAM_ID: team_id}
        if filters is not None:
            data[ApiField.FILTER] = filters
        response = self._api.post("webhooks.list", data)
        results = []
        for item in response.json().get("entities", []):
            meta = item.get("meta", {})
            results.append(WebhookInfo(
                id=item.get(ApiField.ID),
                action=item.get("action"),
                url=item.get(ApiField.URL),
                retries_count=meta.get("retriesCount", 5),
                retries_delay=meta.get("retriesDelay", 10),
                team_id=item.get(ApiField.TEAM_ID),
                created_at=item.get(ApiField.CREATED_AT),
                updated_at=item.get(ApiField.UPDATED_AT),
            ))
        return results

    def get_info_by_id(self, webhook_id: int) -> WebhookInfo:
        """
        Get webhook information by ID.

        :param webhook_id: Webhook ID
        :type webhook_id: int
        :return: Webhook information
        :rtype: WebhookInfo
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            webhook_info = api.webhook.get_info_by_id(webhook_id=456)
        """
        response = self._api.post("webhooks.info", {ApiField.ID: webhook_id})
        info = response.json()
        meta = info.get("meta", {})
        return WebhookInfo(
            id=info.get(ApiField.ID),
            action=info.get("action"),
            url=info.get(ApiField.URL),
            retries_count=meta.get("retriesCount", 5),
            retries_delay=meta.get("retriesDelay", 10),
            team_id=info.get(ApiField.TEAM_ID),
            created_at=info.get(ApiField.CREATED_AT),
            updated_at=info.get(ApiField.UPDATED_AT),
        )

    def create(
        self,
        team_id: int,
        url: str,
        action: str,
        retries_count: int = 5,
        retries_delay: int = 10,
        headers: Optional[Dict] = None,
        follow_redirect: bool = True,
        reject_unauthorized: bool = True,
    ) -> WebhookInfo:
        """
        Create a new webhook.

        :param team_id: Team ID
        :type team_id: int
        :param url: Target URL
        :type url: str
        :param action: Webhook action type (use WebhookAction constants)
        :type action: str
        :param retries_count: Number of retry attempts
        :type retries_count: int
        :param retries_delay: Delay between retries in seconds
        :type retries_delay: int
        :param headers: Custom headers
        :type headers: Dict, optional
        :param follow_redirect: Whether to follow redirects
        :type follow_redirect: bool
        :param reject_unauthorized: Whether to reject unauthorized SSL
        :type reject_unauthorized: bool
        :return: Created webhook information
        :rtype: WebhookInfo
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            webhook = api.webhook.create(
                team_id=123,
                url="https://example.com/webhook",
                action=sly.WebhookAction.LABELING_JOB_COMPLETED,
                retries_count=3
            )
        """
        if headers is None:
            headers = {}

        meta = {
            "retriesCount": retries_count,
            "retriesDelay": retries_delay,
            "headers": headers,
            "followRedirect": follow_redirect,
            "rejectUnauthorized": reject_unauthorized,
        }

        data = {
            ApiField.TEAM_ID: team_id,
            ApiField.URL: url,
            "action": action,
            "meta": meta,
        }

        response = self._api.post("webhooks.add", data)
        webhook_id = response.json()["id"]
        return self.get_info_by_id(webhook_id)

    def update(
        self,
        webhook_id: int,
        url: Optional[str] = None,
        action: Optional[str] = None,
        retries_count: Optional[int] = None,
        retries_delay: Optional[int] = None,
        headers: Optional[Dict] = None,
        follow_redirect: Optional[bool] = None,
        reject_unauthorized: Optional[bool] = None,
    ) -> WebhookInfo:
        """
        Update an existing webhook.

        :param webhook_id: Webhook ID to update
        :type webhook_id: int
        :param url: New target URL
        :type url: str, optional
        :param action: New action type
        :type action: str, optional
        :param retries_count: New number of retry attempts
        :type retries_count: int, optional
        :param retries_delay: New delay between retries
        :type retries_delay: int, optional
        :param headers: New custom headers
        :type headers: Dict, optional
        :param follow_redirect: New follow redirect setting
        :type follow_redirect: bool, optional
        :param reject_unauthorized: New reject unauthorized setting
        :type reject_unauthorized: bool, optional
        :return: Updated webhook information
        :rtype: WebhookInfo
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            updated = api.webhook.update(webhook_id=456, url="https://new.com/webhook")
        """
        data = {ApiField.ID: webhook_id}

        if url is not None:
            data[ApiField.URL] = url
        if action is not None:
            data["action"] = action

        # Build meta object if any meta fields are provided
        meta_fields = {}
        if retries_count is not None:
            meta_fields["retriesCount"] = retries_count
        if retries_delay is not None:
            meta_fields["retriesDelay"] = retries_delay
        if headers is not None:
            meta_fields["headers"] = headers
        if follow_redirect is not None:
            meta_fields["followRedirect"] = follow_redirect
        if reject_unauthorized is not None:
            meta_fields["rejectUnauthorized"] = reject_unauthorized

        if meta_fields:
            data["meta"] = meta_fields

        response = self._api.post("webhooks.update", data)
        webhook_id = response.json()["id"]
        return self.get_info_by_id(webhook_id)

    def remove(self, webhook_id: int) -> None:
        """
        Remove a webhook by ID.

        :param webhook_id: Webhook ID to remove
        :type webhook_id: int
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            api.webhook.remove(webhook_id=456)
        """
        data = {"ids": [webhook_id]}
        self._api.post("webhooks.bulk.remove", data)

    def test(
        self,
        team_id: int,
        webhook_id: Optional[int] = None,
        action: Optional[str] = None,
        payload: Optional[Dict] = None,
    ) -> Dict:
        """
        Send a test event to a webhook.

        :param team_id: Team ID
        :type team_id: int
        :param webhook_id: Webhook ID to test
        :type webhook_id: int, optional
        :param action: Action type to simulate
        :type action: str, optional
        :param payload: Custom payload to send
        :type payload: Dict, optional
        :return: Response from the test
        :rtype: Dict
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            api = sly.Api.from_env()
            response = api.webhook.test(team_id=123, webhook_id=456)
        """
        data = {ApiField.TEAM_ID: team_id}

        if webhook_id is not None:
            data[ApiField.ID] = webhook_id
        if action is not None:
            data["action"] = action
        if payload is not None:
            data["payload"] = payload
        else:
            data["payload"] = {}

        response = self._api.post("webhooks.test", data)
        return response.json()
