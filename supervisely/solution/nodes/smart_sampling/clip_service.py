import threading
import time

from supervisely.api.api import Api
from supervisely.sly_logger import logger
from supervisely.solution.components.empty.node import EmptyNode
from supervisely.solution.engine.models import CLIPServiceStatusMessage


class OpenAIClipServiceNode(EmptyNode):
    """
    Node for OpenAI CLIP service.
    This node is used to interact with the OpenAI CLIP service for image and text embeddings.
    """

    TITLE = "OpenAI CLIP"
    DESCRIPTION = "OpenAI CLIP is a powerful model that can be used to generate embeddings for images in your project. These embeddings can be used for various tasks, such as image similarity search, prompt-based image retrieval. In this application, it is used to create an index and search images based on text prompts or clusters."
    ICON = "mdi mdi-apps"
    ICON_COLOR = "#4CAF50"
    ICON_BG_COLOR = "#E8F5E9"

    APP_SLUG = "supervisely-ecosystem/deploy-clip-as-service"

    def __init__(self, *args, **kwargs):
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
        self.api = Api.from_env()
        self._refresh_interval = 60
        self._stop_autorefresh = False
        self._refresh_thread = None
        self._check_service_status()
        self.start_autorefresh(self._refresh_interval)

    # ------------------------------------------------------------------
    # Handels ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_handles(self):
        return [
            {
                "id": "clip_status",
                "type": "source",
                "position": "right",
                "connectable": True,
            },
        ]

    # ------------------------------------------------------------------
    # Events -----------------------------------------------------------
    # ------------------------------------------------------------------
    # def _available_subscribe_methods(self) -> dict:
    #     """Returns a dictionary of methods that can be used for subscribing to events."""
    #     return {
    #         "clip_status": self._check_service_status,
    #     }
    
    def _available_publish_methods(self):
        return {
            "clip_status": self._check_service_status,
        }

    def start_autorefresh(self, interval: int = 60):
        """
        Starts the auto-refresh mechanism for the node.
        :param interval: Refresh interval in seconds.
        """
        self._refresh_interval = interval
        self._stop_autorefresh = False
        if self._refresh_thread is None:
            self._refresh_thread = threading.Thread(target=self._autorefresh, daemon=True)
        if not self._refresh_thread.is_alive():
            self._refresh_thread.start()

    def stop_autorefresh(self, wait: bool = False):
        self._stop_autorefresh = True
        if wait:
            if self._refresh_thread is not None:
                self._refresh_thread.join()

    def _autorefresh(self):
        """
        Auto-refresh method that periodically updates the node.
        """
        t = time.monotonic()
        while not self._stop_autorefresh:
            if time.monotonic() - t >= self._refresh_interval:
                t = time.monotonic()
                try:
                    self._check_service_status()
                except Exception as e:
                    logger.debug(f"Error during auto-refresh: {e}")
            time.sleep(1)

    def _check_service_status(self) -> CLIPServiceStatusMessage:
        """
        Checks the status of the OpenAI CLIP service.
        This method can be extended to include actual service checks.
        """
        module_id = self.api.app.get_ecosystem_module_id(slug=self.APP_SLUG)
        sessions = self.api.app.get_sessions(
            team_id=1,  # Assuming that CLIP service is deployed in admin team with ID 1
            module_id=module_id,
            with_shared=True,
        )
        if not sessions:
            logger.debug("No active sessions found for OpenAI CLIP service.")
            self.remove_badge_by_key(key="On")
        else:
            logger.debug(f"Active sessions found: {len(sessions)}")
            if len(sessions) > 0:
                logger.debug("OpenAI CLIP service is ready.")
                self.update_badge_by_key(key="On", label="⚡", plain=True)
            else:
                self.remove_badge_by_key(key="On")
                logger.debug("OpenAI CLIP service is not ready yet.")

        return CLIPServiceStatusMessage(is_ready=len(sessions) > 0)
