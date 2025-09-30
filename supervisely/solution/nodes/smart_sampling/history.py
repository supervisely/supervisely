from datetime import datetime
from typing import Any, Dict, List, Optional

from supervisely.api.api import Api
from supervisely.api.image_api import ImageInfo
from supervisely.app.content import DataJson
from supervisely.app.widgets import FastTable, TasksHistory
from supervisely.project.image_transfer_utils import compare_projects
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.solution.components.tasks_history.tasks_history import (
    TasksHistoryWidget,
)
from supervisely.solution.engine.modal_registry import ModalRegistry


class SmartSamplingTasksHistory(TasksHistoryWidget):
    """Tasks history widget specialised for Smart Sampling node."""

    def __init__(
        self,
        api: Optional[Api] = None,
        widget_id: Optional[str] = None,
        project_id: Optional[int] = None,
        dst_project_id: Optional[int] = None,
    ):
        super().__init__(api, widget_id)
        self.project_id = project_id
        self.dst_project_id = dst_project_id

    # ------------------------------------------------------------------
    # --- Table --------------------------------------------------------
    # ------------------------------------------------------------------
    @property
    def table_columns(self) -> List[str]:
        """Header names for the tasks table."""
        if not hasattr(self, "_table_columns"):
            self._table_columns = [
                "Task ID",
                "Mode",
                "Date and Time",
                "Items Count",
                "Settings",
                "Status",
            ]
        return self._table_columns

    @property
    def columns_keys(self) -> List[List[str]]:
        """Mapping between :pyattr:`table_columns` and task dict keys."""
        if not hasattr(self, "_columns_keys"):
            self._columns_keys = [
                ["task_id"],
                ["mode"],
                ["timestamp"],
                ["items_count"],
                ["settings"],
                ["status"],
            ]
        return self._columns_keys

    @property
    def sampled_images(self) -> Dict[int, List[int]]:
        """
        Get sampled images from DataJson.
        :return: dict, sampled images by dataset ID
        """
        self._sampled_images = DataJson()[self.widget_id].get("sampled_images", {})
        return self._sampled_images

    # ------------------------------------------------------------------
    # --- Modal with Preview Gallery -----------------------------------
    # ------------------------------------------------------------------
    def _on_table_row_click(self, clicked_row: FastTable.ClickedRow):
        self.gallery.clean_up()
        if clicked_row.row[5] == "failed":
            return super()._on_table_row_click(clicked_row)
        task_id = clicked_row.row[0]
        sampled_images = self.sampled_images.get(task_id, {})
        if not sampled_images:
            return super()._on_table_row_click(clicked_row)
        ModalRegistry().open_preview(owner_id=self.widget_id)
        self.gallery.loading = True
        ids = [img for imgs in sampled_images.values() for img in imgs]
        infos = self.api.image.get_info_by_id_batch(ids)
        anns = self.api.annotation.download_batch(dataset_id=infos[0].dataset_id, image_ids=ids)
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
    # --- Add Task -----------------------------------------------------
    # ------------------------------------------------------------------
    def update(self):
        """Refresh the table with the current set of tasks."""
        self.table.clear()
        for row in self._get_table_data():
            self.table.insert_row(row)

    def add_task(self, task_id: Optional[str], settings: dict, images_count: int, status: str):
        settings_copy = settings.copy()
        mode = settings_copy.pop("mode", None)
        settings_str = ", ".join(f"{k}={v}" for k, v in settings_copy.items())
        if task_id is None:
            task_id = "N/A"
            status = "failed"
        elif images_count == 0:
            status = "no images sampled"
        history_item = {
            "task_id": task_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mode": mode,
            "settings": settings_str,
            "items_count": images_count,
            "status": status,
        }
        return super().add_task(history_item)

    # ------------------------------------------------------------------
    # --- Differences Helpers ------------------------------------------
    # ------------------------------------------------------------------
    def _filter_diffs(
        self, diffs: Dict[int, List[ImageInfo]], sampled_images: Dict[int, List[int]]
    ) -> Dict[int, List[ImageInfo]]:
        """Filter out already sampled images from the differences."""
        filtered_diffs = {}
        for ds_id, imgs in diffs.items():
            ignore_ids = {img for img in sampled_images.get(ds_id, [])}
            filtered_diffs[ds_id] = [img for img in imgs if img.id not in ignore_ids]
        return filtered_diffs

    def calculate_differences(self) -> Dict[int, List[ImageInfo]]:
        """Calculate the differences between the source and destination projects."""
        diffs = compare_projects(
            api=self.api,
            src_project_id=self.project_id,
            dst_project_id=self.dst_project_id,
        )
        if not diffs:
            return {}

        sampled_images = self.get_all_sampled_images()
        filtered_diffs = self._filter_diffs(diffs, sampled_images)
        return filtered_diffs

    # ------------------------------------------------------------------
    # --- Sampled Images Methods ---------------------------------------
    # ------------------------------------------------------------------
    def _add_sampled_images(
        self,
        task_id: str,
        images: Dict[int, List[ImageInfo]],
    ):
        """
        Save sampled images to DataJson.
        :param task_id: str, Task ID of the sampling task
        :param images: dict, sampled images by dataset ID
        """
        if "sampled_images" not in DataJson()[self.widget_id]:
            DataJson()[self.widget_id]["sampled_images"] = {}
        DataJson()[self.widget_id]["sampled_images"][task_id] = images
        DataJson().send_changes()

    def get_all_sampled_images(self) -> Dict[int, List[int]]:
        """
        Get sampled images from DataJson.
        :return: dict, sampled images by dataset ID
        """
        res = {}
        for _, images in self.sampled_images.items():
            for ds_id, img_list in images.items():
                if ds_id not in res:
                    res[ds_id] = []
                res[ds_id].extend(img_list)
        return res

    def _get_uploaded_ids(self, project_id: int, task_id: int) -> Dict[int, List[int]]:
        """Get the IDs of images uploaded from the project's custom data."""
        project = self.api.project.get_info_by_id(project_id)
        if project is None:
            logger.warning(f"Project with ID {project_id} not found.")
            return {}
        custom_data = project.custom_data or {}
        history = custom_data.get("import_history", {}).get("tasks", [])
        for record in history:
            if record.get("task_id") == task_id:
                break
        else:
            logger.warning(f"No import history found for task ID {task_id}.")
            return {}
        uploaded_ids = {}
        for ds in record.get("datasets", []):
            uploaded_ids[int(ds["id"])] = list(map(int, ds.get("uploaded_images", [])))
        return uploaded_ids
