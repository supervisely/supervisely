from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple, Optional

from supervisely.api.module_api import ApiField, ModuleApi


@dataclass
class WorkflowNode:
    """
    Workflow node - is a target to which a workflow can be applied.
    """

    VALID_TYPES = (
        "task",
        "project",
        "create-project-version",
        "crear-job",
        "creat-queue",
        "team-file",
    )

    NodeTypes = Literal[
        "task",
        "project",
        "create-project-version",
        "crear-job",
        "creat-queue",
        "team-file",
    ]

    def __init__(self, id: int, type: NodeTypes):
        if not isinstance(id, int):
            raise ValueError("id must be an integer")
        if type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}")

        self.id = id
        self.type = type

    def __repr__(self):
        return str(
            {
                "type": self.type,
                "id": self.id,
            }
        )


@dataclass
class WorkflowData:
    """
    Workflow data - is a data that can sent to or received from a workflow node.
    """

    type: Literal[
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
    id: Optional[int]
    meta: Optional[Dict[str, str]]


@dataclass
class WorkflowInfo(NamedTuple):
    """
    NamedTuple WorkflowInfo containing information about Workflow.
    """

    team_id: int
    workspace_id: Optional[int]


class WorkflowApi(ModuleApi):
    """
    Workflow API is a tool for managing workflows in Supervisely.
    """

    def add_transaction(
        self,
        workflow: WorkflowInfo,
        node: WorkflowNode,
        data: WorkflowData,
        data_type: Literal["input", "output"],
    ):
        """
        Add input or output to a workflow node.

        :param workflow: WorkflowInfo - information about the workflow.
        :param node: WorkflowNode - node to which data will be added.
        :param data: WorkflowData - data to be added.
        :param data_type: str - type of data to be added (input or output).
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

        api_endpoint = f"workflow.add-{data_type}"
        payload = {
            ApiField.TEAM_ID: workflow.team_id,
            ApiField.NODE: node,
            ApiField.TYPE: data.type,
        }

        if data.id:
            payload[ApiField.ID] = data.id

        if data.meta:
            payload[ApiField.META] = data.meta

        response = self._api.post(api_endpoint, payload)
        return response
