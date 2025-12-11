import requests

from supervisely import env
from supervisely.api.module_api import ModuleApiBase


class CloudApi(ModuleApiBase):

    def billing_reserve(
        self, user_id: int, items_count: int, cloud_token: str, cloud_action_id: str
    ):
        server_address = env.sly_cloud_server_address()
        method = "/billing/reserve"
        response = requests.post(
            url=server_address + method,
            json={"userId": user_id, "objects": items_count},
            headers={"x-sly-cloud-token": cloud_token, "x-sly-cloud-action-id": cloud_action_id},
        )
        if response.status_code != requests.codes.ok:  # pylint: disable=no-member
            self._api._raise_for_status(response)
        return response.json()

    def billing_withdrawal(
        self,
        user_id: int,
        items_count: int,
        transaction_id: str,
        cloud_token: str,
        cloud_action_id: str,
    ):
        server_address = env.sly_cloud_server_address()
        method = "/billing/withdrawal"
        response = requests.post(
            url=server_address + method,
            json={
                "userId": user_id,
                "objects": items_count,
                "transactionId": transaction_id,
            },
            headers={"x-sly-cloud-token": cloud_token, "x-sly-cloud-action-id": cloud_action_id},
        )
        if response.status_code != requests.codes.ok:  # pylint: disable=no-member
            self._api._raise_for_status(response)
        return response.json()
