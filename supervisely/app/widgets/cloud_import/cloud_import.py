from typing import Callable, List, Optional
from venv import logger

from supervisely.api.api import Api
from supervisely.app.widgets.agent_selector.agent_selector import AgentSelector
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.input.input import Input
from supervisely.app.widgets.field.field import Field
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory
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
        one_at_a_time: bool = True,
        widget_id: Optional[str] = None,
    ):
        """
        :param project_id: ID of the project to import data into.
        :param widget_id: Optional widget ID for the node.
        """
        self.api = Api.from_env()
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)
        self.one_at_a_time = one_at_a_time
        self.workspace_id = self.project.workspace_id
        self.on_start_callbacks = []
        self.on_finish_callbacks = []
        self._init_tasks_history()
        self._create_gui()

        super().__init__(widget_id=widget_id)

    def _update_tasks_history(self):
        tasks = self.tasks_history.get_tasks().copy()
        self.project = self.api.project.get_info_by_id(self.project_id)
        full_history = self.project.custom_data.get("import_history", {}).get("tasks", [])
        history_dict = {item["task_id"]: item for item in full_history}

        for task in tasks:
            task_id = task["id"]
            history_item = history_dict.get(task_id)
            if history_item is None:
                task["dataset_ids"] = ""
                task["timestamp"] = ""
                task["items_count"] = 0
            else:
                datasets = history_item.get("datasets", [])
                task["dataset_ids"] = ", ".join(str(d["id"]) for d in datasets)
                task["timestamp"] = history_item.get("timestamp", "")
                task["items_count"] = history_item.get("items_count", 0)
            self.tasks_history.update_task(task_id, task)
        TasksHistory.update(self.tasks_history)

    def _init_tasks_history(self):
        self.tasks_history = TasksHistory()
        self.tasks_history.table_columns = [
            "Task ID",
            "App Name",
            "Dataset IDs",
            "Created At",
            "Images Count",
            "Status",
        ]
        self.tasks_history.columns_keys = [
            ["id"],
            ["meta", "app", "name"],
            ["dataset_ids"],
            ["created_at"],
            ["items_count"],
            ["status"],
        ]
        self.tasks_history.update = self._update_tasks_history

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
        input_field = Field(
            self.path_input,
            title="Remote Path",
            description="Example: s3://my-bucket/my-folder/my-dataset/img/",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-cloud-upload",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        self.agent_select = AgentSelector(self.project.team_id, compact=True)
        agent_select_field = Field(
            self.agent_select,
            title="Agent",
            description="Select the agent to run the import task.",
            icon=Field.Icon(
                zmdi_class="zmdi zmdi-desktop-mac",
                color_rgb=(21, 101, 192),
                bg_color_rgb=(227, 242, 253),
            ),
        )

        self.status_text = Text("", status="text")
        self.status_text.hide()
        self.run_btn = Button("Run")
        run_btn_cont = Container([self.run_btn], style="align-items: flex-end")
        self.content = Container(
            [text, input_field, agent_select_field, self.status_text, run_btn_cont], gap=20
        )

        # @self.run_btn.click
        # def _on_run_btn_click():
        #     self.run()

    def _validate_path(self, path: str) -> bool:
        """
        Validate a cloud storage path and set error message if validation fails.
        Expected format: provider://bucket-name/path/to/folder
        Example: s3://my-bucket/my-folder/my-dataset/img/
        :param path: Cloud storage path to validate
        :return: True if path is valid, False otherwise
        """
        if path is None or path.strip() == "":
            self.status_text.set("Path cannot be empty", status="error")
            self.status_text.show()
            return False

        path = path.strip()

        if "://" not in path:
            self.status_text.set(
                "Invalid cloud path format. Expected format: provider://bucket-name/path/to/folder",
                status="error",
            )
            self.status_text.show()
            return False

        try:
            provider, rest = path.split("://", 1)
        except ValueError:
            self.status_text.set(
                "Invalid cloud path format. Expected format: provider://bucket-name/path/to/folder",
                status="error",
            )
            self.status_text.show()
            return False

        if not provider:
            self.status_text.set(
                "Cloud provider is required (e.g., s3, gcs, azure)", status="error"
            )
            self.status_text.show()
            return False

        if not rest:
            self.status_text.set("Bucket name and path are required", status="error")
            self.status_text.show()
            return False

        # Check if bucket name exists (first part before /)
        if "/" not in rest:
            self.status_text.set("Path must include bucket name and folder path", status="error")
            self.status_text.show()
            return False

        bucket_name = rest.split("/")[0]
        if not bucket_name:
            self.status_text.set("Bucket name cannot be empty", status="error")
            self.status_text.show()
            return False
        return True

    def run(self, path: Optional[str] = None) -> int:
        self.status_text.hide()
        if path is None:
            path = self.path_input.get_value().strip()

        is_valid_path = self._validate_path(path)
        if not is_valid_path:
            return None
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
        self.status_text.show()
        self.status_text.set("Importing from cloud storage...", status="info")
        for callback in self.on_start_callbacks:
            callback()
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
        task_info = self.api.task.get_info_by_id(session.task_id)
        self.tasks_history.add_task(task_info)

        for callback in self.on_finish_callbacks:
            # if the callback expects arguments, pass the task_id
            if callable(callback):
                if callback.__code__.co_argcount == 0:
                    callback()
                else:
                    callback(session.task_id)

        self.status_text.hide()
        return session.task_id

    def wait_import_completion(self, task_id: int) -> bool:
        """Wait for import task to complete and return status"""
        try:
            self.api.app.wait(task_id, target_status=self.api.task.Status.FINISHED)
            status = self.api.task.get_status(task_id)
            return status == self.api.task.Status.FINISHED
        except Exception as e:
            logger.error(f"Import task {task_id} failed: {str(e)}")
            return False

    @property
    def tasks(self) -> List[int]:
        """
        Collects and returns the import tasks history.
        :return: List of cloud import tasks ids.
        """
        return self.tasks_history.get_tasks().copy()

    def get_json_data(self) -> dict:
        """
        Returns the current data of the Cloud Import widget.
        """
        return {
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
        }

    def get_json_state(self) -> dict:
        """
        Returns the current state of the Cloud Import widget.
        """
        return {}

    def to_html(self) -> str:
        return self.content.to_html()

    def on_finish(self, func: Callable[[], None]) -> None:
        """
        Set a callback function to be called after the import task is finished.
        :param func: Function to call after the import task is finished.
        """
        self.on_finish_callbacks.append(func)
        return func

    def on_start(self, func: Callable[[], None]) -> None:
        """
        Set a callback function to be called before the import task starts.
        :param func: Function to call before the import task starts.
        """
        self.on_start_callbacks.append(func)
        return func
