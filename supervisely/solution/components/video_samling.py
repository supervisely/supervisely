from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.app.widgets.agent_selector.agent_selector import AgentSelector
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.dialog.dialog import Dialog
from supervisely.app.widgets.sampling.sampling import Sampling, SamplingSettings
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory
from supervisely.app.widgets.text.text import Text
from supervisely.solution.base_node import (
    Automation,
    SolutionCard,
    SolutionCardNode,
    SolutionElement,
)


class SamplingAutomation(Automation):
    def __init__(self):
        self.widget = Text("This is a placeholder for the automation widget.")


class VideoSampling(SolutionElement):
    APP_SLUG = "supervisely-ecosystem/turn-video-project-into-images"
    # APP_VERSION = "v3.0.0"
    APP_VERSION = "run-options"
    JOB_ID = "video_sampling_job"

    def __init__(
        self,
        api: Api,
        project_id: int,
        output_project_id: int,
        x: int = 0,
        y: int = 0,
        widget_id: str = None,
    ):
        self.api = api
        self.project_id = project_id
        self.project_info = self.api.project.get_info_by_id(project_id)
        self.output_project_id = output_project_id
        self.automation = SamplingAutomation()
        self.tasks_history = TasksHistory(self.api)
        self.card = self._create_card()
        self.node = SolutionCardNode(x=x, y=y, content=self.card)

        @self.card.click
        def show_run_modal():
            self.run_modal.show()

        self.modals = [
            self.sync_modal,
            self.tasks_modal,
            self.tasks_history.logs_modal,
            self.run_modal,
        ]

        super().__init__(widget_id=widget_id)

    def _sampling_settings_to_app_params(self, sampling_settings: dict) -> dict:
        return {
            "Options": (
                "annotated" if sampling_settings.get(SamplingSettings.ONLY_ANNOTATED) else "all"
            ),
            "sampleResultFrames": True,
            "framesStep": sampling_settings.get(SamplingSettings.STEP, 1),
            "run": True,
        }

    def _run_sampling_app(self):
        agent_id = self.select_agent.get_value()
        module_id = self.api.app.get_ecosystem_module_id(self.APP_SLUG)
        module_info = self.api.app.get_ecosystem_module_info(module_id, version=self.APP_VERSION)

        # Prepare parameters
        params = module_info.get_arguments(videos_project=self.project_id)
        params["allDatasets"] = True
        params["includeNestedDatasets"] = True

        sampling_settings = self.sampling_widget.get_settings()
        params.update(self._sampling_settings_to_app_params(sampling_settings))

        output_project_id = self.output_project_id
        params["outputProjectId"] = output_project_id

        # Start sampling task
        session = self.api.app.start(
            agent_id=agent_id,
            module_id=module_id,
            workspace_id=self.project_info.workspace_id,
            task_name="Samling from Solution",
            params=params,
            app_version=self.APP_VERSION,
            is_branch=True,
        )

        logger.info(f"Video sampling started on agent {agent_id} (task_id: {session.task_id})")
        task_info = self.api.task.get_info_by_id(session.task_id)
        self.tasks_history.add_task(task_info)

        return session.task_id

    @property
    def select_agent(self):
        if not hasattr(self, "_select_agent"):
            self._select_agent = AgentSelector(team_id=self.project_info.team_id, compact=True)
        return self._select_agent

    @property
    def sampling_widget(self):
        if not hasattr(self, "_sampling_widget"):
            self._sampling_widget = Sampling(
                project_id=self.project_id,
                input_selectable=False,
                output_project_id=self.output_project_id,
                output_project_selectable=False,
            )
            self.sampling_widget.preview_container = Container(
                widgets=[
                    self.sampling_widget.preview_container,
                    self.select_agent,
                    self.sampling_widget.run_button_container,
                ]
            )
            self._sampling_widget.run = self._run_sampling_app

        return self._sampling_widget

    @property
    def main_widget(self):
        if not hasattr(self, "_main_widget"):
            self._main_widget = Container(widgets=[self.sampling_widget, self.tasks_button])
        return self._main_widget

    @property
    def sync_modal(self):
        if not hasattr(self, "_sync_modal"):
            self._sync_modal = Dialog(
                title="Automate sampling synchronization",
                content=self.automation.widget,
                size="tiny",
            )
        return self._sync_modal

    @property
    def tasks_modal(self):
        if not hasattr(self, "_tasks_modal"):
            self._tasks_modal = Dialog(title="Sampling tasks history", content=self.tasks_history)
        return self._tasks_modal

    @property
    def run_modal(self):
        if not hasattr(self, "_run_modal"):
            self._run_modal = Dialog(
                title="Sample videos", content=self.sampling_widget, size="tiny"
            )
        return self._run_modal

    def _create_tasks_button(self):
        btn = Button(
            "Import tasks history",
            icon="zmdi zmdi-view-list-alt",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_tasks_dialog():
            self.tasks_history.update()
            self.tasks_modal.show()

        return btn

    @property
    def tasks_button(self):
        if not hasattr(self, "_tasks_button"):
            self._tasks_button = self._create_tasks_button()

        return self._tasks_button

    @property
    def automation_button(self):
        if not hasattr(self, "_automation_button"):
            self._automation_button = self._create_automation_button()

        return self._automation_button

    def _create_automation_button(self):
        btn = Button(
            "Automate",
            icon="zmdi zmdi-flash-auto",
            button_size="mini",
            plain=True,
            button_type="text",
        )

        @btn.click
        def _show_automate_dialog():
            self.sync_modal.show()

        return btn

    def _create_tooltip(self):
        return SolutionCard.Tooltip(
            description="Sample videos from the project and sync them to the images project",
            content=[self.tasks_button, self.automation_button],
        )

    def _create_card(self) -> SolutionCard:
        return SolutionCard(
            title="Sample videos in project",
            tooltip=self._create_tooltip(),
            width=250,
        )
