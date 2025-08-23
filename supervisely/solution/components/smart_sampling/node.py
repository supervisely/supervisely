from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Union

from supervisely.api.api import Api
from supervisely.api.project_api import ProjectInfo
from supervisely.io.env import project_id as env_project_id
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import (
    EmbeddingsStatusMessage,
    ImportFinishedMessage,
    SampleFinishedMessage,
)
from supervisely.solution.utils import get_interval_period

from .automation import SmartSamplingAutomation
from .gui import SamplingMode, SmartSamplingGUI
from .history import SmartSamplingTasksHistory


class SmartSamplingNode(BaseCardNode):
    progress_badge_key = "Sampling"
    title = "Smart Sampling"
    description = (
        "Selects a data sample from the input project and copies it to the labeling project. "
        "Supports various sampling strategies: random, k-means clustering, diversity-based, or using "
        "embeddings precomputed by the “AI Index” node for smarter selection."
    )
    icon = "zmdi zmdi-playlist-plus"
    icon_color = "#1976D2"
    icon_bg_color = "#E3F2FD"

    def __init__(self, project_id: int = None, dst_project: int = None, *args, **kwargs):
        """Node for sampling data from the input project and copying it to the labeling project."""
        self.api = Api.from_env()
        self.project_id = project_id or env_project_id()
        self.dst_project = dst_project or os.getenv("LABELING_PROJECT_ID")
        self.project = self.api.project.get_info_by_id(project_id)

        # --- core blocks --------------------------------------------------------
        self.gui = SmartSamplingGUI(project=self.project, dst_project_id=dst_project)
        self.automation = SmartSamplingAutomation(self.run)
        self.history = SmartSamplingTasksHistory(self.api)

        # --- init node ------------------------------------------------------
        title = kwargs.pop("title", self.title)
        description = kwargs.pop("description", self.description)
        icon = kwargs.pop("icon", self.icon)
        icon_color = kwargs.pop("icon_color", self.icon_color)
        icon_bg_color = kwargs.pop("icon_bg_color", self.icon_bg_color)
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        # --- modals -------------------------------------------------------------
        self.modals = [
            self.gui.modal,
            self.automation.modal,
            self.history.modal,
            self.history.logs_modal,
        ]

        # --- events -------------------------------------------------------------
        self._update_widgets()

        # ! TODO:
        # @self.card.click
        # def on_sampling_setup_btn_click():
        #     """Show the sampling settings modal."""
        #     self.gui.modal.show()
        #     self._check_embeddings_status()

        @self.gui.save_settings_button.click
        def on_save_settings_click():
            self.gui.modal.hide()
            self._update_widgets()

        @self.gui.run_button.click
        def on_run_button_click():
            self.gui.modal.hide()
            self.gui.show_status_text()
            self.gui.set_status_text("Sampling is in progress...", "info")
            self.run()
            self.gui.hide_status_text()

        @self.automation.apply_button.click
        def on_apply_automation_click():
            self.automation.modal.hide()
            self.apply_automation()

        @self.gui.sampling_mode.value_changed
        def on_sampling_mode_change(value: str):
            """Update the sampling settings based on the selected mode."""
            self.gui.collapse_preview()
            self.gui.preview_gallery.clean_up()
            self._check_embeddings_status()

    def _get_tooltip_buttons(self):
        return [self.history.open_modal_button, self.automation.open_modal_button]

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "project_updated",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "embedding_status_response",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
            {
                "id": "sample_finished",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
            # {
            #     "id": "source-2",
            #     "type": "source",
            #     "position": "right",
            #     "connectable": True,
            # },
        ]

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
            self.show_automation_badge()
            self.update_property("Run every", f"{interval} {period}", highlight=True)
        else:
            self.hide_automation_badge()
            self.remove_property_by_key("Run every")

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
            "project_updated": self._update_widgets,
            "embedding_status_response": self._process_embeddings_status_message,
        }

    # callback method (accepts Message object)
    def _update_widgets(
        self, message: Union[ImportFinishedMessage, SampleFinishedMessage] = None
    ) -> None:
        """Update the sampling inputs based on the difference."""
        self.project = self.api.project.get_info_by_id(self.project_id)
        self.items_count = self.project.items_count
        self.gui.set_total_num_text(self.items_count)
        sampling_settings = self.gui.get_settings()

        diff = self.gui.calculate_diff_count()
        self.gui.update_widgets(diff, sampling_settings)

        if diff == 0:
            self.remove_badge_by_key("New Data:")
        else:
            self.update_badge_by_key("New Data:", str(diff), "info")
        self.update_property("New Data:", str(diff))

        mode = sampling_settings.get("mode", SamplingMode.RANDOM.value)
        self.update_property("mode", mode)
        if mode in [SamplingMode.RANDOM.value, SamplingMode.DIVERSE.value]:
            sample_size = sampling_settings.get("sample_size", 0)
            self.update_property("Sample size", str(sample_size))
        elif mode == SamplingMode.AI_SEARCH.value:
            prompt = sampling_settings.get("prompt", "")
            limit = sampling_settings.get("limit", 0)
            threshold = sampling_settings.get("threshold", 0.05)
            self.update_property("Prompt", prompt)
            self.update_property("Limit", str(limit))
            self.update_property("Threshold", f"{threshold:.2f}")

    # ------------------------------------------------------------------
    # Methods --------------------------------------------------------
    # ------------------------------------------------------------------
    def _check_embeddings_status(self):
        """Send message to check embeddings status."""
        if self.gui.sampling_mode.get_value() == SamplingMode.RANDOM.value:
            self.gui.preview_button.enable()
            self.gui.run_button.enable()
        else:
            self.gui.preview_button.disable()
            self.gui.run_button.disable()

    def _process_embeddings_status_message(self, message: EmbeddingsStatusMessage) -> None:
        """Process the embeddings status message."""
        is_ready = message.status
        if is_ready or self.gui.sampling_mode.get_value() == SamplingMode.RANDOM.value:
            self.gui.preview_button.enable()
            self.gui.run_button.enable()
        else:
            self.gui.preview_button.disable()
            self.gui.run_button.disable()

    # ------------------------------------------------------------------
    # Run --------------------------------------------------------------
    # ------------------------------------------------------------------

    # publish_event (may send Message object)
    def run(self) -> SampleFinishedMessage:
        self.show_in_progress_badge("Sampling")
        res = self.gui.run()
        src, dst, images_count = res
        self.hide_in_progress_badge("Sampling")
        self._update_widgets()
        return SampleFinishedMessage(success=True, src=src, dst=dst, items_count=images_count)
