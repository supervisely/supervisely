from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple, Optional

from supervisely.api.module_api import ApiField, ModuleApi


@dataclass
class WorkflowNode:
    """
    Workflow node - is a target to which a workflow can be applied.

    :param node_id: int - id of the node.
    :param node_type: str - type of the node.
                Can be one of the following:
                    "task", "project", "create-project-version", "create-job", "create-queue", "team-file".
    """

    VALID_TYPES = (
        "task",
        "project",
        "create-project-version",
        "create-job",
        "create-queue",
        "team-file",
    )

    NodeTypes = Literal[
        "task",
        "project",
        "create-project-version",
        "create-job",
        "create-queue",
        "team-file",
    ]

    def __init__(self, node_id: int, node_type: NodeTypes):
        if not isinstance(node_id, int):
            raise ValueError("id must be an integer")
        if node_type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}")

        self.node_id = node_id
        self.node_type = node_type

    def __repr__(self):
        return str(
            {
                "type": self.node_type,
                "id": self.node_id,
            }
        )


@dataclass
class WorkflowData:
    """
    Workflow data - is a data that can sent to or received from a workflow node.

    :param data_type: str - type of data.
                Can be one of the following:
                    "project", "dataset", "task", "job", "project-version", "file", "folder", "model-weight", "app-ui".
    :param data_id: Optional[int] - id of the data. If not provided, the data type must be "folder" or remote "file" and "meta" must be provided.
    :param meta: Optional[Dict[str, str]] - metadata that helps to identify the data properly.
    """

    data_type: Literal[
        "project",
        "dataset",
        "task",
        "job",
        "project-version",
        "file",
        "folder",
        "model-weight",
        "app-ui",
    ]
    data_id: Optional[int] = None
    meta: Optional[Dict[str, str]] = None


@dataclass
class WorkflowInfo(NamedTuple):
    """
    NamedTuple WorkflowInfo containing information about Workflow.

    :param team_id: int - team id.
    :param workspace_id: Optional[int] - workspace id.
    """

    team_id: int
    workspace_id: Optional[int] = None


class WorkflowApi(ModuleApi):
    """
    Workflow API is a tool for managing workflows in Supervisely.
    """

    def add_transaction(
        self,
        workflow: WorkflowInfo,
        node: WorkflowNode,
        data: WorkflowData,
        transaction_type: Literal["input", "output"],
    ):
        """
        Add input or output to a workflow node.

        :param workflow: WorkflowInfo - information about the workflow.
        :param node: WorkflowNode - node to which data will be added.
        :param data: WorkflowData - data to be added.
        :param transaction_type: Literal["input", "output"] - type of transaction.
        :return: dict - response from the API.

        Example:
        ```python
                from supervisely import Api, WorkflowData, WorkflowInfo, WorkflowNode

                api = Api.from_env()

                workflow = WorkflowInfo(team_id=1)
                node = WorkflowNode(id=1, type="task")
                data = WorkflowData(type="folder", meta={"slyFolder": "/path/to/team-files/folder"})
                response = api.workflow.add_transaction(workflow, node, data, "input")
        ```

        """

        api_endpoint = f"workflow.add-{transaction_type}"
        payload = {
            ApiField.TEAM_ID: workflow.team_id,
            ApiField.NODE: node,
            ApiField.TYPE: data.data_type,
        }

        if data.data_id:
            payload[ApiField.ID] = data.data_id

        if data.meta:
            payload[ApiField.META] = data.meta

        response = self._api.post(api_endpoint, payload)
        return response
