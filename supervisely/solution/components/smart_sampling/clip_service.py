import threading
import time

from supervisely.api.api import Api
from supervisely.app.widgets import Icons
from supervisely.io.env import team_id
from supervisely.sly_logger import logger
from supervisely.solution.components.empty.node import EmptyNode


class OpenAIClipServiceNode(EmptyNode):
    """
    Node for OpenAI CLIP service.
    This node is used to interact with the OpenAI CLIP service for image and text embeddings.
    """

    APP_SLUG = "supervisely-ecosystem/deploy-clip-as-service"

    def __init__(self, x: int = 0, y: int = 0, *args, **kwargs):
        super().__init__(
            title="OpenAI CLIP",
            description="OpenAI CLIP is a powerful model that can be used to generate embeddings for images in your project. These embeddings can be used for various tasks, such as image similarity search, prompt-based image retrieval. In this application, it is used to create an index and search images based on text prompts or clusters.",
            width=150,
            icon=Icons(class_name="zmdi zmdi-apps", color="#4CAF50", bg_color="#E8F5E9"),
            tooltip_position="left",
            x=x,
            y=y,
            *args,
            **kwargs,
        )
        self.api = Api.from_env()
        self._refresh_interval = 60
        self._stop_autorefresh = False
        self._refresh_thread = None
        # self.start_autorefresh(30)

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

    def _check_service_status(self):
        """
        Checks the status of the OpenAI CLIP service.
        This method can be extended to include actual service checks.
        """
        module_id = self.api.app.get_ecosystem_module_id(slug=self.APP_SLUG)
        sessions = self.api.app.get_sessions(
            team_id=team_id(),
            module_id=module_id,
            with_shared=True,
        )
        if not sessions:
            logger.debug("No active sessions found for OpenAI CLIP service.")
            self.node.remove_badge_by_key(key="On")
        else:
            logger.debug(f"Active sessions found: {len(sessions)}")
            is_ready = False
            for session in sessions:
                is_ready = self.api.task.is_ready(session.task_id)
                break
            if is_ready:
                logger.debug("OpenAI CLIP service is ready.")
                self.node.update_badge_by_key(key="On", label="⚡", plain=True)
            else:
                self.node.remove_badge_by_key(key="On")
                logger.debug("OpenAI CLIP service is not ready yet.")
