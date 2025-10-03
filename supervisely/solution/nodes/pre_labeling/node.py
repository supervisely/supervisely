import datetime
import time
from threading import Thread
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from supervisely.api.api import Api
from supervisely.app.widgets import Button
from supervisely.sly_logger import logger
from supervisely.solution.base_node import BaseCardNode
from supervisely.solution.engine.models import ModelDeployMessage, SampleFinishedMessage
from supervisely.solution.nodes.pre_labeling.gui import PreLabelingGUI
from supervisely.solution.nodes.pre_labeling.history import PreLabelingTasksHistory


class PreLabelingNode(BaseCardNode):
    TITLE = "Pre-labeling"
    DESCRIPTION = "Automatically generate predictions for images using the deployed custom model."
    ICON = "mdi mdi-auto-fix"
    ICON_COLOR = "#1976D2"
    ICON_BG_COLOR = "#E3F2FD"

    def __init__(
        self,
        tooltip_position: Literal["left", "right"] = "right",
        *args,
        **kwargs,
    ):
        self.api = Api.from_env()
        self.tooltip_position = tooltip_position

        # Initialize components
        self.history = PreLabelingTasksHistory()
        self.gui = PreLabelingGUI(api=self.api)
        self.modal_content = self.gui.content

        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", None)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            *args,
            **kwargs,
        )

        # Widget event handlers
        @self.gui.enable_switch.value_changed
        def on_enable_switch_change(value: bool):
            self._save_settings(enabled=value)

        @self.click
        def on_card_click():
            self.gui.open_modal()

        # Load settings and update properties
        self._update_properties(self.gui.enable_switch.is_switched())

    def _get_tooltip_buttons(self):
        if not hasattr(self, "_tooltip_buttons"):
            self._tooltip_buttons = [self.history.open_modal_button, self.run_btn]
        return self._tooltip_buttons

    # ------------------------------------------------------------------
    # Handles ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "start_pre_labeling",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "model_deployed",
                "type": "target",
                "position": "right",
                "connectable": True,
            },
            {
                "id": "pre_labeling_finished",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_subscribe_methods(self):
        return {
            "start_pre_labeling": self._process_incoming_message,
            "model_deployed": self._process_incoming_message,
        }

    def _available_publish_methods(self):
        return {
            "pre_labeling_finished": self._send_output_message,
        }

    def _process_incoming_message(self, msg: Union[SampleFinishedMessage, ModelDeployMessage]):
        if isinstance(msg, ModelDeployMessage):
            self._set_deployed_model(msg.session_id)
        elif isinstance(msg, SampleFinishedMessage):
            if not msg.dst or not isinstance(msg.dst, dict):
                logger.warning("No destination images provided in the message.")
                return
            images = []
            for imgs in msg.dst.values():
                images.extend(imgs)
            if len(images) == 0:
                logger.info("No images to process for pre-labeling.")
                return
            # self.run_async(images=images)
            self.run(images=images)

    def _send_output_message(self, items_count: int) -> SampleFinishedMessage:
        return SampleFinishedMessage(items_count=items_count, success=True, src={}, dst={})

    # ------------------------------------------------------------------
    # Methods ----------------------------------------------------------
    # ------------------------------------------------------------------

    @property
    def run_btn(self) -> Button:
        if not hasattr(self, "_run_btn"):
            self._run_btn = Button(
                "Run manually",
                icon="zmdi zmdi-play",
                button_size="mini",
                plain=True,
                button_type="text",
            )

        return self._run_btn

    def _update_properties(self, enabled: bool):
        """Update node properties based on current settings."""
        status = "enabled" if enabled else "disabled"
        self.update_property("Pre-labeling", status, highlight=enabled)

        if enabled:
            self.show_automation_badge()
        else:
            self.hide_automation_badge()

        # Update processed images count
        processed_count = len(self._get_all_processed_images())
        if processed_count > 0:
            self.update_property("Processed images", str(processed_count))

    def _save_settings(self, enabled: Optional[bool] = None):
        """Save pre-labeling settings."""
        if enabled is None:
            enabled = self.gui.enable_switch.is_switched()

        self.gui.save_settings(enabled)
        self._update_properties(enabled)

    def _load_settings(self):
        """Load pre-labeling settings."""
        self.gui.load_settings()
        self._update_properties()

    def _is_enabled(self) -> bool:
        """Check if pre-labeling is enabled."""
        return self.gui.enable_switch.is_switched()

    def _set_deployed_model(self, session_id: int):
        """Set reference to the deployed custom model."""
        if not isinstance(session_id, int):
            raise ValueError("Model must be an integer session ID.")
        self.gui.set_model_session_id(session_id)
        logger.info("Pre-labeling: Custom model reference set")

    def _get_all_processed_images(self) -> List[int]:
        """Get all processed image IDs from all tasks."""
        return self.gui._get_processed_images()

    def _run(self, images: List[int]) -> Optional[Dict[str, Any]]:
        if not self._is_enabled():
            logger.info("Pre-labeling is disabled, skipping...")
            return None

        if self.gui.model is None:
            logger.warning("No model selected, cannot perform pre-labeling")
            return None

        start_time = time.time()
        started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_id = len(self.history.get_tasks()) + 1

        self.show_in_progress_badge("Pre-labeling")

        try:
            self.gui.run(images=images)

            last_processed_images = self.gui._get_last_processed_images()
            self.gui.update_preview_gallery(last_processed_images)

            # Create task record
            duration = time.time() - start_time
            task_record = {
                "task_id": task_id,
                "started_at": started_at,
                "images_count": len(images),
                "status": "Success",
                "duration": f"{duration:.2f}s",
            }

            self.history.add_task(task_record)
            self._update_properties(True)

            logger.info(
                f"Pre-labeling completed: {len(images)} images processed in {duration:.2f}s"
            )

            return {
                "task_id": task_id,
                "images": self.gui._get_last_processed_images(),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Pre-labeling failed: {e}", exc_info=True)

            # Create error task record
            task_record = {
                "task_id": task_id,
                "started_at": started_at,
                "images_count": 0,
                "status": "Error",
                "duration": f"{time.time() - start_time:.2f}s",
            }
            self.history.add_task(task_record)

            return {"task_id": task_id, "status": "error", "error": str(e)}

        finally:
            self.hide_in_progress_badge("Pre-labeling")

    def run(self, images: List[int]) -> Optional[Dict[str, Any]]:
        """
        Run pre-labeling on the provided images.

        Args:
            images: List of image IDs to process.
        Returns:
            Dictionary with processing results or None if disabled/failed
        """
        return self._run(images)

    # run asynchronously
    def run_async(self, images: List[int]) -> None:
        """
        Run pre-labeling asynchronously on the provided images.

        Args:
            images: List of image IDs to process.
        """
        thread = Thread(target=self._run, args=(images,))
        thread.start()
