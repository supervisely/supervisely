import threading
import time
from datetime import datetime
from typing import Callable, Dict

from supervisely.api.api import Api
from supervisely.io.env import project_id as env_project_id
from supervisely.sly_logger import logger
from supervisely.solution.components.empty.node import EmptyNode
from supervisely.solution.engine.models import (
    CLIPServiceStatusMessage,
    EmbeddingsStatusMessage,
)


class AiIndexNode(EmptyNode):
    """
    Node for OpenAI CLIP service.
    This node is used to interact with the OpenAI CLIP service for image and text embeddings.
    """

    TITLE = "AI Index"
    DESCRIPTION = "AI Search Index is a powerful tool that allows you to search for images in your dataset using AI models. It provides a quick and efficient way to find similar images based on visual features. You can use it in Smart Sampling node to select images for labeling based on specified prompt."
    ICON = "mdi mdi-image-search"
    ICON_COLOR = "#4CAF50"
    ICON_BG_COLOR = "#E8F5E9"
    APP_SLUG = "supervisely-ecosystem/deploy-clip-as-service"

    def __init__(self, project_id: int = None, *args, **kwargs):

        title = kwargs.pop("title", self.TITLE)
        description = kwargs.pop("description", self.DESCRIPTION)
        icon = kwargs.pop("icon", self.ICON)
        icon_color = kwargs.pop("icon_color", self.ICON_COLOR)
        icon_bg_color = kwargs.pop("icon_bg_color", self.ICON_BG_COLOR)
        super().__init__(
            title=title,
            description=description,
            icon=icon,
            icon_color=icon_color,
            icon_bg_color=icon_bg_color,
            width=150,
            tooltip_position="left",
            *args,
            **kwargs,
        )
        self.project_id = project_id or env_project_id()
        self._refresh_interval = 20
        self._stop_autorefresh = False
        self._refresh_thread = None
        self.api = Api.from_env()
        self.check_embeddings_status()

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "clip_status",
                "type": "target",
                "position": "left",
                "connectable": True,
            },
            {
                "id": "embedding_status_request",
                "type": "target",
                "position": "top",
                "connectable": True,
            },
            {
                "id": "embedding_status_response",
                "type": "source",
                "position": "bottom",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _available_publish_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "embedding_status_response": self.send_embeddings_status_message,
        }

    def _available_subscribe_methods(self) -> Dict[str, Callable]:
        """Returns a dictionary of methods that can be used for publishing events."""
        return {
            "embedding_status_request": self.check_embeddings_status,
            "clip_status": self.process_message_from_clip_service,
        }

    def process_message_from_clip_service(self, message: CLIPServiceStatusMessage) -> None:
        if message.is_ready:
            self.check_embeddings_status()

    def check_embeddings_status(self) -> EmbeddingsStatusMessage:
        """Check that project embeddings are enabled, not in progress, and up to date."""
        is_ready = False
        try:
            is_ready = (
                self._is_embeddings_enabled()
                and not self._is_embeddings_in_progress()
                and self._check_embeddings_updated_at()
            )
            if not is_ready:
                self.start_autorefresh()
        except Exception as e:
            logger.error(f"Error checking AI Index status: {repr(e)}")
        finally:
            self.send_embeddings_status_message(is_ready)

    def send_embeddings_status_message(self, is_ready: bool) -> EmbeddingsStatusMessage:
        if is_ready:
            self.update_badge_by_key(key="Up to date", label="⚡", plain=True)
            self.update_property("Embeddings", "Up to date", highlight=True)
        else:
            self.remove_badge_by_key(key="Up to date")
            self.remove_property_by_key("Embeddings")
        return EmbeddingsStatusMessage(status=is_ready)

    # ------------------------------------------------------------------
    # Private methods --------------------------------------------------
    # ------------------------------------------------------------------
    def _is_embeddings_enabled(self):
        is_embeddings_enabled = self.api.project.is_embeddings_enabled(self.project_id)
        if not is_embeddings_enabled:
            logger.info(f"Embeddings are not enabled for project {self.project_id}. Enabling...")
            self.api.project.enable_embeddings(self.project_id)
            embeddings_updated_at = self.api.project.get_embeddings_updated_at(self.project_id)
            if embeddings_updated_at is None:
                logger.info(
                    f"Embeddings are not updated for project {self.project_id}. Updating..."
                )
                self.api.project.calculate_embeddings(self.project_id)
            return False
        return True

    def _check_embeddings_updated_at(self):
        """Check if embeddings are up to date."""
        embeddings_updated_at = self.api.project.get_embeddings_updated_at(self.project_id)
        if embeddings_updated_at is None:
            logger.info(f"Embeddings are not up to date for project {self.project_id}. Updating...")
            self.api.project.calculate_embeddings(self.project_id)
            return False
        project_info = self.api.project.get_info_by_id(self.project_id)
        embeddings_updated_at = datetime.fromisoformat(embeddings_updated_at.replace("Z", "+00:00"))
        project_updated_at = datetime.fromisoformat(project_info.updated_at.replace("Z", "+00:00"))
        if embeddings_updated_at < project_updated_at:
            logger.info(f"Embeddings are not up to date for project {self.project_id}. Updating...")
            self.api.project.calculate_embeddings(self.project_id)
            return False
        logger.debug(f"Embeddings are up to date for project {self.project_id}.")
        return True

    def _is_embeddings_in_progress(self):
        """Check if embeddings calculation is in progress."""
        is_in_progress = self.api.project.get_embeddings_in_progress(self.project_id)
        if is_in_progress:
            logger.info(f"Embeddings calculation is in progress for project {self.project_id}.")
        return is_in_progress

    def _check_status_and_send_message(self):
        """
        Check the AI Index status and send the status message.
        This method is called periodically to ensure the AI Index is up to date.
        """
        try:
            is_ready = self.check_embeddings_status()
            if is_ready:  # send message only if embeddings are ready
                self.stop_autorefresh()
                self.send_embeddings_status_message(is_ready)
        except Exception as e:
            logger.error(f"Error during auto-refresh: {repr(e)}")

    # ------------------------------------------------------------------
    # Auto-refresh -----------------------------------------------------
    # ------------------------------------------------------------------
    def _autorefresh(self):
        """
        Auto-refresh method that periodically checks the AI Index status.
        """
        t = time.monotonic()
        while not self._stop_autorefresh:
            if time.monotonic() - t >= self._refresh_interval:
                t = time.monotonic()
                try:
                    self._check_status_and_send_message()
                except Exception as e:
                    logger.debug(f"Error during auto-refresh: {e}")
            time.sleep(1)

    def start_autorefresh(self, refresh_interval: int = 20):
        """
        Start the auto-refresh thread to periodically check the AI Index status.
        :param refresh_interval: Interval in seconds for checking the status.
        """
        self._refresh_interval = refresh_interval
        self._stop_autorefresh = False
        if self._refresh_thread is None:
            self._refresh_thread = threading.Thread(target=self._autorefresh, daemon=True)
        if not self._refresh_thread.is_alive():
            self._refresh_thread.start()
            logger.debug("AI Index auto-refresh started.")

    def stop_autorefresh(self, wait: bool = False):
        self._stop_autorefresh = True
        if wait:
            if self._refresh_thread is not None:
                try:
                    self._refresh_thread.join()
                    logger.debug("AI Index auto-refresh stopped.")
                except Exception as e:
                    logger.error(f"Error stopping AI Index auto-refresh: {repr(e)}")
