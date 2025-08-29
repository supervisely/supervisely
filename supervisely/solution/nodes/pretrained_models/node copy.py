from typing import Callable, List, Optional, Tuple, Union

import supervisely.io.env as sly_env
from supervisely import ProjectMeta
from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.api.entities_collection_api import CollectionTypeFilter
from supervisely.api.project_api import ProjectInfo
from supervisely.api.task_api import TaskApi
from supervisely.app.content import DataJson
from supervisely.app.exceptions import show_dialog
from supervisely.app.widgets import (
    Button,
    Checkbox,
    Container,
    Dialog,
    Empty,
    Icons,
    InputNumber,
    NewExperiment,
    Select,
    SolutionCard,
    Text,
    Widget,
)
from supervisely.nn.task_type import TaskType
from supervisely.project.project_type import ProjectType
from supervisely.sly_logger import logger
from supervisely.solution.base_node import Automation, SolutionCardNode, SolutionElement
from supervisely.solution.components.tasks_history import SolutionTasksHistory
from supervisely.solution.utils import get_interval_period


class TrainAutomation(Automation):
    TRAIN_JOB_ID = "train_model_job"
    CHECK_STATUS_JOB_ID = "check_train_status_job"

    def __init__(self):
        self.apply_btn = Button("Apply", plain=True)
        self.apply_text = Text("", status="text", color="gray")
        self.widget = self._create_widget()
        self.job_id = self.widget.widget_id
        self.func = None
        super().__init__()

    def _create_widget(self):
        self.enabled_checkbox = Checkbox(content="Run every", checked=False)
        self.num_input = InputNumber(min=1, value=60, debounce=1000, controls=False, size="mini")
        self.num_input.disable()
        self.period_select = Select(
            [
                Select.Item("min", "minutes"),
                Select.Item("h", "hours"),
                Select.Item("d", "days"),
            ],
            size="mini",
        )
        self.period_select.disable()
        automate_cont = Container(
            [
                self.enabled_checkbox,
                self.num_input,
                self.period_select,
                Empty(),
            ],
            direction="horizontal",
            gap=3,
            fractions=[1, 1, 1, 1],
            style="align-items: center",
        )
        apply_btn = Container([self.apply_btn], style="align-items: flex-end")
        self.apply_text.set("Run training first to save settings.", "warning")

        @self.enabled_checkbox.value_changed
        def on_automate_checkbox_change(is_checked: bool) -> None:
            if is_checked:
                self.num_input.enable()
                self.period_select.enable()
            else:
                self.num_input.disable()
                self.period_select.disable()

        return Container([self.apply_text, automate_cont, apply_btn])

    def apply(self, func: Callable, sec: int, job_id: str, *args) -> None:
        self.scheduler.add_job(func, sec, job_id, True, *args)
        logger.info(f"Scheduled model training job with ID {job_id} every {sec} seconds.")

    def remove(self, job_id: str) -> None:
        if self.scheduler.is_job_scheduled(job_id):
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed scheduled model training job with ID {job_id}.")

    def get_automation_details(self) -> Tuple[bool, str, int, int]:
        """
        Get the automation details from the widget.
        :return: Tuple with (enabled, period, interval, seconds)
        """
        enabled = self.enabled_checkbox.is_checked()
        period = self.period_select.get_value()
        interval = self.num_input.get_value()

        if not enabled:
            return False, None, None, None

        if period == "h":
            sec = interval * 60 * 60
        elif period == "d":
            sec = interval * 60 * 60 * 24
        else:
            sec = interval * 60
        if sec == 0:
            return False, None, None, None
        return enabled, period, interval, sec


class TrainTasksHistory(SolutionTasksHistory):
    def __init__(self, api: Api, title: str = "Tasks History"):
        super().__init__(api, title)
        self.tasks_history._table_columns = [
            "Task ID",
            "Model ID",
            "Started At",
            "Status",
            "Agent ID",
            "Classes Count",
            "Images Count",
        ]
        self.tasks_history._columns_keys = [
            ["task_info", "id"],
            ["model_id"],
            ["task_info", "created_at"],
            ["status"],
            ["agent_id"],
            ["classes_count"],
            ["images_count"],
        ]

    def update_task_status(self, task_id: int, status: str):
        tasks = self.tasks_history.get_tasks()
        task = None
        for row in tasks:
            if row["id"] == task_id:
                task = row
                row["status"] = status
                self.tasks_history.update_task(task_id=task_id, task=task)
                return
        raise KeyError(f"Task with ID {task_id} not found in the task history.")


class BaseTrainGUI(Widget):
    cv_task: TaskType = TaskType.OBJECT_DETECTION
    frameworks: Optional[List[str]] = None

    def __init__(
        self,
        api: Api,
        project: Union[ProjectInfo, int],
        team_id: Optional[int] = None,
        workspace_id: Optional[int] = None,
        frameworks: Optional[List[str]] = None,
        widget_id: Optional[str] = None,
    ):
        self.api = api
        self.project = (
            project
            if isinstance(project, ProjectInfo)
            else self.api.project.get_info_by_id(project)
        )
        self.workspace_id = workspace_id or self.project.workspace_id
        self.team_id = team_id or self.project.team_id
        self.frameworks = frameworks
        super().__init__(widget_id=widget_id)
        self.content = self._init_gui()

    def _init_gui(self) -> NewExperiment:
        train_collections, val_collections = self._get_train_val_collections()
        split_mode = "collections" if train_collections and val_collections else "random"

        project_meta = ProjectMeta.from_json(self.api.project.get_meta(self.project.id))
        classes = [obj_cls.name for obj_cls in project_meta.obj_classes]

        content = NewExperiment(
            team_id=self.team_id,
            workspace_id=self.workspace_id,
            project_id=self.project.id,
            classes=classes,
            step=5,  # 5 - start with model selection
            filter_projects_by_workspace=True,
            project_types=[ProjectType.IMAGES],
            cv_task=self.cv_task,
            selected_frameworks=self.frameworks,
            train_val_split_mode=split_mode,  # only collections?
            train_collections=train_collections,
            val_collections=val_collections,
            # gui selectors disabled
            cv_task_selection_disabled=True,  # 1 - cv task selection
            project_selection_disabled=True,  # 2 - project selection
            classes_selection_disabled=False,  # 3 - classes selection
            train_val_split_selection_disabled=True,  # 4 - train/val split selection
            model_selection_disabled=False,  # 5 - model selection
            evaluation_selection_disabled=False,  # 9 - evaluation selection
            speed_test_selection_disabled=False,  # 9 - speed test selection
            framework_selection_disabled=self.frameworks is not None,
            architecture_selection_disabled=True,
        )

        @content.visible_changed
        def _on_visible_changed(visible: bool):
            print(f"NewExperiment visibility changed: {visible}")

        return content

    def _get_train_val_collections(self) -> Tuple[List[int], List[int]]:
        if self.project.type != ProjectType.IMAGES.value:
            return [], []
        train_collections, val_collections = [], []
        all_collections = self.api.entities_collection.get_list(self.project.id)
        for collection in all_collections:
            if collection.name == "All_train":
                train_collections.append(collection.id)
            elif collection.name == "All_val":
                val_collections.append(collection.id)

        return train_collections, val_collections

    def get_json_data(self):
        return {}

    def get_json_state(self):
        return {}


class BaseTrainNode(SolutionElement):
    gui_class = BaseTrainGUI
    title = "Train Model"
    description = "Train a model on the selected project. The model will be trained on the training collection and validated on the validation collection. If collections are not set, the model will be trained on random split of the project images."

    def __init__(
        self,
        api: Api,
        project: Union[ProjectInfo, int],
        x: int = 0,
        y: int = 0,
        icon: Optional[Icons] = None,
        *args,
        **kwargs,
    ):
        self.icon = icon
        self.width = 250
        self.project_id = project.id if isinstance(project, ProjectInfo) else project

        self.api = api
        self.tasks_history = TrainTasksHistory(self.api, title="Train Tasks History")
        self.main_widget = self.gui_class(api=api, project=self.project_id)
        self.automation = TrainAutomation()

        self.card = self._create_card()
        self.node = SolutionCardNode(content=self.card, x=x, y=y)
        self.modals = [
            self.tasks_history.tasks_modal,
            self.tasks_history.logs_modal,
            self.main_widget.content,
            self.automation_modal,
        ]

        self._train_settings = None
        self._previous_task_id = None

        self._train_started_cb = []
        self._train_finished_cb = []
        super().__init__(*args, **kwargs)

        @self.card.click
        def _on_card_click():
            # if not self.main_widget.content.visible:
            self.main_widget.content.visible = True

        @self.main_widget.content.app_started
        def _on_app_started(app_id: int, model_id: int, task_id: int):
            self.main_widget.content.visible = False
            self._previous_task_id = task_id
            self._save_train_settings()

            # add task to tasks history
            task_info = self.api.task.get_info_by_id(task_id)

            train_collection = self.main_widget.content.train_collections
            train_collection = train_collection[0] if train_collection else None
            val_collection = self.main_widget.content.val_collections
            val_collection = val_collection[0] if val_collection else None

            images_count = "N/A"
            if train_collection and val_collection:
                train_imgs = self.api.entities_collection.get_items(
                    train_collection, CollectionTypeFilter.DEFAULT
                )
                val_imgs = self.api.entities_collection.get_items(
                    val_collection, CollectionTypeFilter.DEFAULT
                )
                images_count = f"train: {len(train_imgs)}, val: {len(val_imgs)}"
            task = {
                "task_info": task_info,
                "model_id": self.main_widget.content.model_id,
                "status": "started",
                "agent_id": self.main_widget.content.agent_id,
                "classes_count": len(self.main_widget.content.classes),
                "images_count": images_count,
            }
            self.tasks_history.add_task(task=task)

            self.automation.apply(
                self._check_train_progress,
                10,
                self.automation.CHECK_STATUS_JOB_ID,
                task_id,
            )
            for cb in self._train_started_cb:
                if not callable(cb):
                    logger.error(f"Train started callback {cb} is not callable.")
                    continue
                try:
                    if cb.__code__.co_argcount == 3:
                        cb(app_id, model_id, task_id)
                    elif cb.__code__.co_argcount == 1:
                        cb(task_id)
                    else:
                        cb()
                except Exception as e:
                    logger.error(f"Error in train started callback: {e}")

        @self.automation_apply_button.click
        def on_automation_apply_button_click():
            self.automation_modal.hide()
            enabled, _, _, sec = self.automation.get_automation_details()
            self.show_automation_info(enabled, sec)
            self.automation.apply(
                self._run_automated_task, sec, self.automation.CHECK_STATUS_JOB_ID
            )

    def _save_train_settings(self):
        """
        Extract training configuration from the embedded NewExperiment widget and store it
        inside the node so that it can be reused later
        """
        try:
            self._train_settings = self.main_widget.content.get_train_settings()
            logger.info("Training settings saved.")
        except Exception as e:
            logger.warning(f"Failed to save training settings: {e}")

    @property
    def train_settings(self):
        """Returns the most recently saved training settings, or None if nothing saved."""
        return self._train_settings

    def _check_train_progress(self, task_id: int):
        # @ TODO: get train status from the task (fix send request on web progress status message)
        # train_status = self.api.task.send_request(task_id, "train_status", {})
        # print(f"Train status: {train_status}")

        task_info = self.api.task.get_info_by_id(task_id)
        if task_info is not None:
            if task_info["status"] == TaskApi.Status.ERROR.value:
                self.card.update_badge_by_key(key="Status", label="Failed", badge_type="error")
                self.automation.remove(self.automation.CHECK_STATUS_JOB_ID)
            elif task_info["status"] == TaskApi.Status.CONSUMED.value:
                self.card.update_badge_by_key(key="Status", label="Consumed", badge_type="warning")
            elif task_info["status"] == TaskApi.Status.QUEUED.value:
                self.card.update_badge_by_key(key="Status", label="Queued", badge_type="warning")
            elif task_info["status"] in [
                TaskApi.Status.STOPPED.value,
                TaskApi.Status.TERMINATING.value,
            ]:
                self.card.update_badge_by_key(key="Status", label="Stopped", badge_type="warning")
                self.automation.remove(self.automation.CHECK_STATUS_JOB_ID)
            elif task_info["status"] == TaskApi.Status.FINISHED.value:
                self.card.update_badge_by_key(key="Status", label="Finished", badge_type="success")
                for cb in self._train_finished_cb:
                    if not callable(cb):
                        logger.error(f"Train finished callback {cb} is not callable.")
                        continue
                    try:
                        if cb.__code__.co_argcount == 1:
                            cb(task_id)
                        else:
                            cb()
                    except Exception as e:
                        logger.error(f"Error in train finished callback: {e}")
                self.automation.remove(self.automation.CHECK_STATUS_JOB_ID)
            else:
                self.card.update_badge_by_key(key="Status", label="Training...", badge_type="info")
        else:
            logger.error(f"Task info is not found for task_id: {task_id}")

    def _create_card(self) -> SolutionCard:
        return SolutionCard(
            title=self.title,
            tooltip=self._create_tooltip(),
            width=self.width,
            icon=self.icon,
            tooltip_position="right",
        )

    def _create_tooltip(self) -> SolutionCard.Tooltip:
        return SolutionCard.Tooltip(
            description=self.description,
            content=[self.tasks_button, self.open_session_button, self.automation_button],
        )

    @property
    def tasks_button(self) -> Button:
        if not hasattr(self, "_tasks_button"):
            self._tasks_button = self._create_tasks_button()
        return self._tasks_button

    def _create_tasks_button(self) -> Button:
        btn = Button(
            text="Tasks History",
            icon="zmdi zmdi-view-list",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_tasks_dialog():
            self.tasks_history.tasks_history.update()
            self.tasks_history.tasks_modal.show()

        return btn

    @property
    def open_session_button(self) -> Button:
        if not hasattr(self, "_open_session_button"):
            self._open_session_button = self._create_open_session_button()
        return self._open_session_button

    def _create_open_session_button(self) -> Button:
        return Button(
            text="Open Running Session",
            icon="zmdi zmdi-open-in-new",
            button_size="mini",
            plain=True,
            button_type="text",
            link=self.session_link,
        )

    @property
    def session_link(self) -> str:
        return self._session_link if hasattr(self, "_session_link") else ""

    @session_link.setter
    def session_link(self, value: str):
        if not hasattr(self, "_session_link"):
            setattr(self, "_session_link", value)
        else:
            self._session_link = value
        self.open_session_button.link = value

    def set_collection_ids(
        self,
        train_collection_id: Optional[int] = None,
        val_collection_id: Optional[int] = None,
    ):
        """
        Set the collection IDs for training and validation collections.
        """
        if train_collection_id is not None:
            self.main_widget.content.train_collections = [train_collection_id]
        if val_collection_id is not None:
            self.main_widget.content.val_collections = [val_collection_id]

    def set_classes(self, classes: Union[List[str], List[ObjClass]]):
        """
        Set the classes for the training session.
        """
        if isinstance(classes, list) and all(isinstance(cls, str) for cls in classes):
            self.main_widget.content.classes = classes
        elif isinstance(classes, list) and all(isinstance(cls, ObjClass) for cls in classes):
            self.main_widget.content.classes = [c.name for c in classes]
        else:
            raise ValueError("Classes must be a list of strings or ObjClass instances.")

    def on_train_started(self, fn: Callable) -> Callable:
        """
        Register a callback function to be called when the training starts.
        """
        self._train_started_cb.append(fn)
        return fn

    def on_train_finished(self, fn: Callable) -> Callable:
        """
        Register a callback function to be called when the training finishes.
        """
        self._train_finished_cb.append(fn)
        return fn

    def check_train_finished(self, task_id: int) -> bool:
        """
        Check if the training task has finished.
        """
        # todo: Implement the logic to check train task status.
        pass

    # Automation
    @property
    def automation_apply_button(self) -> Button:
        """Get the apply training button"""
        return self.automation.apply_btn

    @property
    def automation_modal(self):
        if not hasattr(self, "_automation_modal"):
            self._automation_modal = self._create_automation_modal()
        return self._automation_modal

    @property
    def automation_button(self):
        if not hasattr(self, "_automation_button"):
            self._automation_button = self._create_automation_button()
        return self._automation_button

    def _create_automation_button(self):
        btn = Button(
            "Automate training",
            icon="zmdi zmdi-flash-auto",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_automate_dialog():
            if self._previous_task_id is None:
                self.automation.apply_text.set("Run training first to save settings.", "warning")
                self.automation.apply_btn.disable()
            else:
                self.automation.apply_text.set(
                    "Schedule automatic model training on the training data. <br> <strong>Note:</strong> The settings from the last training session will be used.",
                    "text",
                )
                self.automation.apply_btn.enable()
            self.automation_modal.show()

        return btn

    def _create_automation_modal(self) -> Dialog:
        return Dialog(
            title="Automate Training",
            content=self.automation.widget,
            size="tiny",
        )

    def show_automation_info(self, enabled, sec):
        period, interval = get_interval_period(sec)
        if enabled is True:
            self.node.show_automation_badge()
            self.card.update_property("Run every", f"{interval} {period}", highlight=True)
        else:
            self.node.hide_automation_badge()
            self.card.remove_property_by_key("Run every")

    def _run_automated_task(self) -> None:
        if self._previous_task_id is None:
            logger.error("No previous task id found. Run training first to save settings.")
            return

        try:
            task_id = self._start_automated_training()
            if task_id is None:
                logger.error(f"Task info is not found for task_id: {self._previous_task_id}")
                return
        except Exception as e:
            logger.error(f"Failed to run automated training task: {e}")
            return
        self.automation.apply(
            self._check_train_progress, 10, self.automation.CHECK_STATUS_JOB_ID, task_id
        )

    def _start_automated_training(self) -> Optional[int]:
        task_info = self.api.task.get_info_by_id(self._previous_task_id)
        if task_info is None:
            return None

        description = f"Automated training run from Solution app. Task ID: {self._previous_task_id}"
        agent_id = task_info["agentId"]
        workspace_id = task_info["workspaceId"]
        params = task_info["meta"]["params"]

        app_info = task_info["meta"]["app"]
        module_id = app_info["moduleId"]
        is_branch = app_info["isBranch"]
        version = app_info["version"]

        session_info = self.api.app.start(
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=workspace_id,
            description=description,
            params=params,
            app_version=version,
            is_branch=is_branch,
            task_name=description,
        )
        return session_info.task_id

    # -------------------------------------- #


# Framework classes


class RTDETRv2TrainGUI(BaseTrainGUI):
    _PREDEFINED_FRAMEWORKS = ["RT-DETRv2"]

    def __init__(
        self,
        api: Api,
        project: Union[ProjectInfo, int],
        workspace_id: Optional[int] = None,
        team_id: Optional[int] = None,
        widget_id: Optional[str] = None,
    ):
        super().__init__(
            api=api,
            project=project,
            workspace_id=workspace_id,
            team_id=team_id,
            frameworks=self._PREDEFINED_FRAMEWORKS,
            widget_id=widget_id,
        )


class RTDETRv2TrainNode(BaseTrainNode):
    gui_class = RTDETRv2TrainGUI
    title = "Train RT-DETRv2"
    description = (
        "Train an RT-DETRv2 model on the selected project using the last saved training "
        "settings. The framework is fixed to RT-DETRv2."
    )


class YOLOTrainGUI(BaseTrainGUI):
    _PREDEFINED_FRAMEWORKS = ["YOLO"]

    def __init__(
        self,
        api: Api,
        project: Union[ProjectInfo, int],
        workspace_id: Optional[int] = None,
        team_id: Optional[int] = None,
        widget_id: Optional[str] = None,
    ):
        super().__init__(
            api=api,
            project=project,
            workspace_id=workspace_id,
            team_id=team_id,
            frameworks=self._PREDEFINED_FRAMEWORKS,
            widget_id=widget_id,
        )


class YOLOTrainNode(BaseTrainNode):
    gui_class = YOLOTrainGUI
    title = "Train YOLO"
    description = (
        "Train YOLO model on the selected project using the last saved training "
        "settings. The framework is fixed to YOLO."
    )


class DEIMTrainGUI(BaseTrainGUI):
    _PREDEFINED_FRAMEWORKS = ["DEIM"]

    def __init__(
        self,
        api: Api,
        project: Union[ProjectInfo, int],
        workspace_id: Optional[int] = None,
        team_id: Optional[int] = None,
        widget_id: Optional[str] = None,
    ):
        super().__init__(
            api=api,
            project=project,
            workspace_id=workspace_id,
            team_id=team_id,
            frameworks=self._PREDEFINED_FRAMEWORKS,
            widget_id=widget_id,
        )


class DEIMTrainNode(BaseTrainNode):
    gui_class = DEIMTrainGUI
    title = "Train DEIM"
    description = (
        "Train DEIM model on the selected project using the last saved training "
        "settings. The framework is fixed to DEIM."
    )


# -------------------------------------- #
