from typing import List, Optional
from venv import logger

from supervisely.api.api import Api
from supervisely.app.widgets.agent_selector.agent_selector import AgentSelector
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.input.input import Input
from supervisely.app.widgets.text.text import Text
from supervisely.app.widgets.widget import Widget


class CloudImport(Widget):
    """
    CloudImport node for importing data from cloud storage to a project.
    This node allows users to import data from a specified path in cloud storage
    and manage import tasks.
    """

    APP_SLUG = "e9b5a1d81aa98072cd77b402fdc122d7/cloud-storage-data-synchronizer"
    JOB_ID = "cloud_import_job"

    def __init__(
        self,
        project_id: int,
        widget_id: Optional[str] = None,
    ):
        """
        :param project_id: ID of the project to import data into.
        :param widget_id: Optional widget ID for the node.
        """
        self.api = Api.from_env()
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)
        self.workspace_id = self.project.workspace_id
        self._tasks = []
        self._create_gui()

        super().__init__(widget_id=widget_id)

    def _create_gui(self):
        """
        Initialize the widgets for the Cloud Import node.
        This method sets up the UI components for importing data from cloud storage.
        """
        text = Text(
            "Select the folder in the Cloud Storage to import data from. Only new items will be imported.",
            status="text",
            color="gray",
        )
        self.path_input = Input(placeholder="provider://bucket-name/path/to/folder")
        self.agent_select = AgentSelector(self.project.team_id, compact=True)
        self.run_btn = Button("Run", plain=True)
        run_btn_cont = Container([self.run_btn], style="align-items: flex-end")
        self.content = Container([text, self.path_input, run_btn_cont])

        @self.run_btn.click
        def _on_run_btn_click():
            self.run()

    def to_html(self):
        return self.content.to_html()

    def run(self) -> int:
        path = self.path_input.get_value().strip()
        agent_id = self.agent_select.get_value()
        return self._run(path, agent_id)

    def _run(self, path: str, agent_id: int) -> int:
        """
        Import data from cloud storage to input project

        :param path: Path to the folder in the Cloud Storage (e.g., "provider://bucket-name/path/to/folder")
        :type path: str
        :return: Task ID of the import task
        :rtype: int
        """
        logger.info(f"Starting import from cloud storage: {path}")

        # Get the module ID for importing from cloud
        module_id = self.api.app.get_ecosystem_module_id(self.APP_SLUG)
        module_info = self.api.app.get_ecosystem_module_info(module_id)

        # Prepare parameters for import
        params = module_info.get_arguments(images_project=self.project_id)
        params["slyFolder"] = path

        # Start import task
        session = self.api.app.start(
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=self.workspace_id,
            task_name="Import from Cloud Storage",
            params=params,
        )

        logger.info(f"Cloud import started on agent {agent_id} (task_id: {session.task_id})")
        self._tasks.append(session.task_id)
        self.update_data()

        return session.task_id

    def wait_import_completion(self, task_id: int) -> bool:
        """Wait for import task to complete and return status"""
        try:
            self.api.app.wait(task_id, target_status=self.api.task.Status.FINISHED)
            return True
        except Exception as e:
            logger.error(f"Import task {task_id} failed: {str(e)}")
            return False

    @property
    def tasks(self) -> List[int]:
        """
        Collects and returns the import tasks history.
        :return: List of cloud import tasks ids.
        """
        return self._tasks

    def _get_table_data(self) -> List[List]:
        """
        Collects and returns the import tasks history as a list of lists.
        """

        self.project = self.api.project.get_info_by_id(self.project_id)
        full_history = self.project.custom_data.get("import_history", {}).get("tasks", [])
        history_dict = {item["task_id"]: item for item in full_history}

        data = []
        for task_id in self.tasks:
            history_item = history_dict.get(task_id)
            if history_item is None:
                data.append([task_id, "", "", "", 0, "failed"])
                continue
            if history_item.get("slug") != self.APP_SLUG:
                logger.warning(
                    f"Import history item with task_id {task_id} does not match the slug {self.APP_SLUG}. Skipping."
                )
                continue
            datasets = history_item.get("datasets", [])
            ds_ids = ", ".join(str(d["id"]) for d in datasets)
            status = history_item.get("status")
            if status == "started":
                status = "ok"
            row = [
                history_item.get("task_id"),
                history_item.get("app", {}).get("name", ""),
                ds_ids,
                history_item.get("timestamp"),
                history_item.get("items_count"),
                status,
            ]
            data.append(row)

        return data

    def get_json_data(self) -> dict:
        """
        Returns the current data of the Cloud Import widget.
        """
        return {
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "tasks": self._tasks,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Cloud Import widget.
        """
        return {}
