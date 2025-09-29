from typing import Any, Dict, List, Optional

from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.app.content import DataJson
from supervisely.solution.components import TasksHistoryWidget
from supervisely.sly_logger import logger


class MoveLabeledTasksHistory(TasksHistoryWidget):
    """Tasks history widget specialised for MoveLabeled node."""

    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "Started At",
                "Images Count",
                "Status",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["id"],
                ["startedAt"],
                ["images_count"],
                ["status"],
            ]
        return self._columns_keys

    # ------------------------------------------------------------------
    # --- Sampled Images Methods ---------------------------------------
    # ------------------------------------------------------------------
    @property
    def moved_images(self) -> Dict[int, List[int]]:
        """
        Get moved images from DataJson.
        :return: dict, moved images by dataset ID
        """
        self._moved_images = DataJson()[self.widget_id].get("moved_images", {})
        return self._moved_images

    def _add_moved_images(
        self,
        task_id: str,
        images: List[ImageInfo],
    ):
        """
        Save moved images to DataJson.
        :param task_id: str, Task ID of the moving task
        :param images: list, moved images
        """
        if "moved_images" not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id]["moved_images"] = {}
        DataJson()[self.widget_id]["moved_images"][task_id] = images
        DataJson().send_changes()

    def get_all_moved_images(self) -> List[int]:
        """
        Get moved images from DataJson.
        :return: list, moved images IDs
        """
        res = []
        for _, images in self.moved_images.items():
            res.extend(images)
        return res

    def _get_uploaded_ids(self, project_id: int, task_id: int) -> List[int]:
        """Get the IDs of images uploaded from the project's custom data."""
        project = self.api.project.get_info_by_id(project_id)
        if project is None:
            logger.warning(f"Project with ID {project_id} not found.")
            return []
        custom_data = project.custom_data or {}
        history = custom_data.get("import_history", {}).get("tasks", [])
        for record in history:
            if record.get("task_id") == task_id:
                break
        else:
            logger.warning(f"No import history found for task ID {task_id}.")
            return []
        uploaded_ids = []
        for ds in record.get("datasets", []):
            uploaded_ids.extend(list(map(int, ds.get("uploaded_images", []))))
        return uploaded_ids
