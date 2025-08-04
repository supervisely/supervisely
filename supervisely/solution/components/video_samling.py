from typing import Tuple

from supervisely._utils import logger
from supervisely.api.api import Api
from supervisely.app.widgets.agent_selector.agent_selector import AgentSelector
from supervisely.app.widgets.button.button import Button
from supervisely.app.widgets.container.container import Container
from supervisely.app.widgets.dialog.dialog import Dialog
from supervisely.app.widgets.notification_box.notification_box import NotificationBox
from supervisely.app.widgets.sampling.sampling import Sampling, SamplingSettings
from supervisely.app.widgets.solution_card.solution_card import SolutionCard
from supervisely.app.widgets.tasks_history.tasks_history import TasksHistory
from supervisely.app.widgets.text.text import Text
from supervisely.solution.components.base.automation import AutomationWidget
from supervisely.solution.components.base.card import SolutionCardNode
from supervisely.solution.components.base.node import SolutionElement


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
        self.tasks_history = TasksHistory(self.api)
        self.card = self._create_card()
        self.node = SolutionCardNode(x=x, y=y, content=self.card)
        self.automation = AutomationWidget(description="Automate sampling", func=lambda: self.run())
        self.sampling_in_progress = False

        @self.card.click
        def show_run_modal():
            self.run_modal.show()

        @self.automation.on_apply
        def on_apply():
            is_enabled = self.automation.is_enabled()
            if is_enabled:
                self.sampling_widget.settings_container.disable()
                self.sampling_widget.run_button.disable()
                self.select_agent_container.disable()
            else:
                self.sampling_widget.settings_container.enable()
                self.sampling_widget.run_button.enable()
                self.select_agent_container.enable()
            self.sync_modal.hide()
            self.run_modal.hide()
            self.update_automation_details()

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

    def run(self):
        if self.sampling_in_progress:
            logger.warning("Sampling is already in progress. Please wait until it completes.")
            return
        self.sampling_in_progress = True
        self.samping_in_progress_notification.show()
        self._sampling_widget.run_button.loading = True
        try:
            agent_id = self.select_agent.get_value()
            module_id = self.api.app.get_ecosystem_module_id(self.APP_SLUG)
            module_info = self.api.app.get_ecosystem_module_info(
                module_id, version=self.APP_VERSION
            )

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
                task_name="Sampling from Solution",
                params=params,
                app_version=self.APP_VERSION,
                is_branch=True,
            )

            logger.info(f"Video sampling started on agent {agent_id} (task_id: {session.task_id})")
            task_info = self.api.task.get_info_by_id(session.task_id)
            self.tasks_history.add_task(task_info)

            self.api.task.wait(id=session.task_id, target_status=self.api.task.Status.TERMINATING)
            project_info = self.api.project.get_info_by_id(output_project_id)
            self.sampling_widget.output_project_preview.set(project_info)
            return session.task_id
        finally:
            self.samping_in_progress_notification.hide()
            self._sampling_widget.run_button.loading = False
            self.sampling_in_progress = False

    @property
    def samping_in_progress_notification(self):
        if not hasattr(self, "_samping_in_progress_notification"):
            self._samping_in_progress_notification = NotificationBox(
                title="Sampling in progress",
                description="Please wait until the sampling is completed.",
            )
        return self._samping_in_progress_notification

    @property
    def select_agent(self):
        if not hasattr(self, "_select_agent"):
            self._select_agent = AgentSelector(team_id=self.project_info.team_id, compact=True)
        return self._select_agent

    @property
    def select_agent_container(self):
        if not hasattr(self, "_select_agent_container"):
            self._select_agent_container = Container(
                widgets=[
                    Text(
                        '<i class="zmdi zmdi-storage" style="padding-right: 10px; padding-top: 20px; color: rgb(0, 154, 255);"></i><b>Select Agent</b>'
                    ),
                    Container(
                        widgets=[self.select_agent], style="padding-left: 21px; padding-top: 10px;"
                    ),
                ],
                gap=0,
                style="padding-top: 10px;",
            )
        return self._select_agent_container

    @property
    def sampling_widget(self):
        if not hasattr(self, "_sampling_widget"):
            self._sampling_widget = Sampling(
                project_id=self.project_id,
                input_selectable=False,
                output_project_id=self.output_project_id,
                output_project_selectable=False,
                width="auto",
            )
            self._sampling_widget.run_button.size = "small"
            self._sampling_widget.run_button.plain = True

            self._sampling_widget.preview_container._widgets = [
                self._sampling_widget.preview_field,
                self.select_agent_container,
                self._sampling_widget.run_button_container,
                self.samping_in_progress_notification,
            ]

            self.samping_in_progress_notification.hide()
            self._sampling_widget.run = lambda: self.run()
            self._sampling_widget.nested_datasets_checkbox.hide()

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
                title="Sample videos",
                content=Container(widgets=[self.sampling_widget, self.automation.widget]),
                size="tiny",
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

    def _show_automation_badge(self) -> None:
        self._update_automation_badge(True)

    def _hide_automation_badge(self) -> None:
        self._update_automation_badge(False)

    def _update_automation_badge(self, enable: bool) -> None:
        for idx, prop in enumerate(self.card.badges):
            if prop["on_hover"] == "Automation":
                if enable:
                    pass  # already enabled
                else:
                    self.card.remove_badge(idx)
                return

        if enable:  # if not found
            self.card.add_badge(
                self.card.Badge(
                    label="⚡",
                    on_hover="Automation",
                    badge_type="warning",
                    plain=True,
                )
            )

    def update_automation_details(self) -> Tuple[int, str, int, str]:
        sec, interval, period = self.automation.get_automation_details()
        # self.sync_modal.hide()
        if sec is not None:
            if self.card is not None:
                self.card.update_property(
                    "Sync", "Every {} {}".format(interval, period), highlight=True
                )
                logger.info(f"Added job to synchronize from Cloud Storage every {sec} sec")
                self._show_automation_badge()
        else:
            if self.card is not None:
                self.card.remove_property_by_key("Sync")
                self.card.remove_property_by_key("Path")
                # g.session.importer.unschedule_cloud_import()
                self._hide_automation_badge()

        return sec, interval, period
