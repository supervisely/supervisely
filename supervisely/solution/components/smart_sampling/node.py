from __future__ import annotations
from typing import Callable, Optional
from supervisely.api.api import Api

from supervisely.api.project_api import ProjectInfo
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.utils import get_interval_period

from .automation import SmartSamplingAutomation
from .gui import SmartSamplingGUI, SamplingMode
from .history import SmartSamplingTasksHistory


class SmartSamplingNode(SolutionElement):
    progress_badge_key = "Sampling"

    def __init__(
        self, api: Api, project_id: int, dst_project: int, x: int, y: int, *args, **kwargs
    ):
        """Node for sampling data from the input project and copying it to the labeling project."""
        self.api = api
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)

        # --- core blocks --------------------------------------------------------
        self.gui = SmartSamplingGUI(project=self.project, dst_project_id=dst_project)
        self.automation = SmartSamplingAutomation(self.run)
        self.tasks_history = SmartSamplingTasksHistory(self.api)  # , widget_id) job_id?
        self.card = self._build_card(
            title="Smart Sampling",
            tooltip_description="Selects a data sample from the input project and copies it to the labeling project. Supports various sampling strategies: random, k-means clustering, diversity-based, or using embeddings precomputed by the “AI Index” node for smarter selection.",
            buttons=[self.tasks_history.open_modal_button, self.automation.open_modal_button],
        )
        self.node = SolutionCardNode(content=self.card, x=x, y=y)

        # --- modals -------------------------------------------------------------
        self.modals = [
            self.gui.modal,
            self.automation.modal,
            self.tasks_history.modal,
            self.tasks_history.logs_modal,
        ]

        # --- callbacks ----------------------------------------------------------
        self.on_start_callbacks = []
        self.on_finish_callbacks = []

        # --- events -------------------------------------------------------------
        self.update_widgets()

        @self.card.click
        def on_sampling_setup_btn_click():
            """Show the sampling settings modal."""
            self.gui.modal.show()

        @self.gui.save_settings_button.click
        def on_save_settings_click():
            self.gui.modal.hide()
            self.update_widgets()

        @self.gui.run_button.click
        def on_run_button_click():
            self.gui.status_text.show()
            self.gui.status_text.set("Sampling is in progress...", status="info")
            self.run()
            self.gui.status_text.hide()

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Automation -------------------------------------------------------
    # ------------------------------------------------------------------
    def apply_automation(self):
        enabled, _, _, sec = self.automation.get_details()
        self.show_automation_info(enabled, sec)
        self.automation.apply()

    def show_automation_info(self, enabled, sec):
        period, interval = get_interval_period(sec)
        if enabled is True:
            self.node.show_automation_badge()
            self.card.update_property("Run every", f"{interval} {period}", highlight=True)
        else:
            self.node.hide_automation_badge()
            self.card.remove_property_by_key("Run every")

    def update_automation_widgets(self):
        enabled, _, _, sec = self.automation.get_details()
        self.automation.save_details(enabled, sec)
        self.show_automation_info(enabled, sec)

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def update_widgets(
        self,
        diff: Optional[int] = None,
        sampling_settings: Optional[dict] = None,
        updated_project_info: Optional[ProjectInfo] = None,
    ):
        """Update the sampling inputs based on the difference."""
        if updated_project_info:
            self.project = updated_project_info
            self.items_count = self.project.items_count
            self.gui.total_num_text.text = f"{self.items_count} images"

        if diff is None:
            diff = self.gui.calculate_diff_count()

        self.gui.update_widgets(diff, sampling_settings)

        if diff == 0:
            self.card.remove_badge_by_key("Difference:")
        else:
            self.card.update_badge_by_key("Difference:", str(diff), "info")
        self.card.update_property("Difference:", str(diff))

        sampling_settings = sampling_settings or self.gui.get_settings()
        mode = sampling_settings.get("mode", SamplingMode.RANDOM.value)
        self.card.update_property("mode", mode)
        if mode in [SamplingMode.RANDOM.value, SamplingMode.DIVERSE.value]:
            sample_size = sampling_settings.get("sample_size", 0)
            self.card.update_property("Sample size", str(sample_size))
        elif mode == SamplingMode.AI_SEARCH.value:
            prompt = sampling_settings.get("prompt", "")
            limit = sampling_settings.get("limit", 0)
            threshold = sampling_settings.get("threshold", 0.05)
            self.card.update_property("Prompt", prompt)
            self.card.update_property("Limit", str(limit))
            self.card.update_property("Threshold", f"{threshold:.2f}")

    # ------------------------------------------------------------------
    # Callbacks --------------------------------------------------------
    # ------------------------------------------------------------------
    def on_start(self, func: Callable):
        """
        Decorator to register a callback function to be called when the sampling starts.
        :param func: Callable, function to be called on start
        :return: Callable, the same function
        """
        self.on_start_callbacks.append(func)
        return func

    def on_finish(self, func: Callable):
        """
        Decorator to register a callback function to be called when the sampling finishes.
        :param func: Callable, function to be called on finish
        :return: Callable, the same function
        """
        self.on_finish_callbacks.append(func)
        return func

    # ------------------------------------------------------------------
    # Run --------------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self):
        self.node.show_in_progress_badge("Sampling")
        for callback in self.on_start_callbacks:
            callback()
        res = self.gui.run()
        for callback in self.on_finish_callbacks:
            if callable(callback):
                if callback.__code__.co_argcount == 0:
                    callback()
                else:
                    callback(res)
        self.node.hide_in_progress_badge("Sampling")
