from typing import Any, Dict, List, Optional

from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.app.content import DataJson
from supervisely.app.widgets.fast_table.fast_table import FastTable
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.solution.components import TasksHistoryWidget
from supervisely.solution.engine.modal_registry import ModalRegistry


class MoveLabeledTasksHistory(TasksHistoryWidget):
    """Tasks history widget specialised for MoveLabeled node."""

    def __init__(self, project_id: int, *args, **kwargs):
        """
        Initialize the MoveLabeledTasksHistory widget.

        :param project_id: int, ID of the destination project where labeled data is moved.
        """
        self.project_id = project_id
        super().__init__(*args, **kwargs)

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
    # --- Modal with Preview Gallery -----------------------------------
    # ------------------------------------------------------------------
    def _on_table_row_click(self, clicked_row: FastTable.ClickedRow):
        self.gallery.clean_up()
        if clicked_row.row[5] == "failed":
            return super()._on_table_row_click(clicked_row)
        task_id = clicked_row.row[0]
        moved_images = self.moved_images.get(task_id, {})
        if not moved_images:
            return super()._on_table_row_click(clicked_row)
        ModalRegistry().open_preview(owner_id=self.widget_id)
        self.gallery.loading = True
        infos = self.api.image.get_info_by_id_batch(moved_images)
        anns = self.api.annotation.download_batch(
            dataset_id=infos[0].dataset_id, image_ids=moved_images
        )
        meta = ProjectMeta.from_json(self.api.project.get_meta(self.project_id))

        for idx, (img, ann) in enumerate(zip(infos, anns)):
            self.gallery.append(
                image_url=img.full_storage_url,
                annotation_info=ann,
                title=img.name,
                column_index=idx % 3,
                project_meta=meta,
                call_update=idx == len(infos) - 1,
            )
        self.gallery.loading = False

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
        images: List[int],
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
