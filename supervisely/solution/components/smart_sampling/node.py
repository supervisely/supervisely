from __future__ import annotations

from typing import Callable, Dict, Optional, Union

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.solution.base_node import SolutionCardNode, SolutionElement
from supervisely.solution.engine.models import (
    ImportFinishedMessage,
    SampleFinishedMessage,
)
from supervisely.solution.utils import get_interval_period

from .automation import SmartSamplingAutomation
from .gui import SamplingMode, SmartSamplingGUI
from .history import SmartSamplingTasksHistory


class SmartSamplingNode(SolutionElement):
    progress_badge_key = "Sampling"

    def __init__(self, project_id: int, dst_project: int, x: int, y: int, *args, **kwargs):
        """Node for sampling data from the input project and copying it to the labeling project."""
        self.api = Api.from_env()
        self.project_id = project_id
        self.project = self.api.project.get_info_by_id(project_id)

        # --- core blocks --------------------------------------------------------
        super().__init__(*args, **kwargs)
        self.gui = SmartSamplingGUI(project=self.project, dst_project_id=dst_project)
        self.automation = SmartSamplingAutomation(self.run)
        self.tasks_history = SmartSamplingTasksHistory(self.api)  # , widget_id) job_id?
        self.card = self._build_card(
            title="Smart Sampling",
            tooltip_description="Selects a data sample from the input project and copies it to the labeling project. Supports various sampling strategies: random, k-means clustering, diversity-based, or using embeddings precomputed by the “AI Index” node for smarter selection.",
            buttons=[self.tasks_history.open_modal_button, self.automation.open_modal_button],
            icon="zmdi zmdi-playlist-plus",
            icon_color="#1976D2",
            icon_bg_color="#E3F2FD",
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
            self.gui.modal.hide()
            self.gui.status_text.show()
            self.gui.status_text.set("Sampling is in progress...", status="info")
            self.run()
            self.gui.status_text.hide()

        @self.automation.apply_button.click
        def on_apply_automation_click():
            self.automation.modal.hide()
            self.apply_automation()

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
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "sample_finished": self.run,
        }

    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for subscribing to events."""
        return {
            "import_finished": self.update_widgets,
            "sample_finished": self.update_widgets,
        }

    # callback method (accepts Message object)
    def update_widgets(
        self, message: Union[ImportFinishedMessage, SampleFinishedMessage] = None
    ) -> None:
        """Update the sampling inputs based on the difference."""
        updated_project_info = self.api.project.get_info_by_id(self.project_id)
        self.project = updated_project_info
        self.items_count = self.project.items_count
        self.gui.total_num_text.text = f"{self.items_count} images"
        sampling_settings = self.gui.get_settings()

        diff = self.gui.calculate_diff_count()
        self.gui.update_widgets(diff, sampling_settings)

        if diff == 0:
            self.card.remove_badge_by_key("Difference:")
        else:
            self.card.update_badge_by_key("Difference:", str(diff), "info")
        self.card.update_property("Difference:", str(diff))

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

    # publish_event (may send Message object)
    def run(self) -> SampleFinishedMessage:
        self.node.show_in_progress_badge("Sampling")
        for callback in self.on_start_callbacks:
            callback()
        res = self.gui.run()
        src, dst, images_count = res
        for callback in self.on_finish_callbacks:
            if callable(callback):
                if callback.__code__.co_argcount == 0:
                    callback()
                else:
                    callback(res)
        self.node.hide_in_progress_badge("Sampling")
        self.update_widgets()
        return SampleFinishedMessage(success=True, src=src, dst=dst, items_count=images_count)
