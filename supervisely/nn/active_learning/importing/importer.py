from typing import TYPE_CHECKING, Dict, List, Optional, Union

from supervisely.nn.active_learning.utils.constants import (
    AUTOIMPORT_SLUG,
    CLOUD_IMPORT_SLUG,
)

if TYPE_CHECKING:
    from supervisely.nn.active_learning.session import ActiveLearningSession

import supervisely.nn.active_learning.importing.from_cloud as cloud_import
import supervisely.sly_logger as logger


class ActiveLearningImporter:
    def __init__(self, al_session):
        self.al_session: ActiveLearningSession = al_session
        self.api = al_session.api
        self.project_id = al_session.project_id
        self.workspace_id = al_session.workspace_id
        self.team_id = al_session.team_id
        self.state = al_session.state

    def import_from_cloud(self, path: str) -> int:
        """Import data from cloud storage to input project"""
        return cloud_import.import_from_cloud(self.al_session, path)

    def wait_import_completion(self, task_id: int) -> bool:
        """Wait for import task to complete and return status"""
        try:
            self.api.app.wait(task_id, target_status=self.api.task.Status.FINISHED)
            return True
        except Exception as e:
            logger.error(f"Import task {task_id} failed: {str(e)}")
            return False

    def schedule_cloud_import(self, path: str, interval: int) -> str:
        """
        Schedule a cloud import task

        Args:
            path (str): Path to the data in cloud storage.
            interval (int): Interval in seconds for the scheduled task.
        """
        return cloud_import.schedule_cloud_import(self.al_session, path, interval)

    def unschedule_cloud_import(self) -> bool:
        """
        Unschedule the cloud import task
        """
        return cloud_import.unschedule_cloud_import(self)

    def get_differences_count(self) -> int:
        src_datasets = self.api.dataset.get_list(self.project_id, recursive=True)
        sampled_items = self.al_session.state.get_sampled_images()

        total_differences = 0
        for ds_info in src_datasets:
            total_differences += ds_info.items_count
            total_differences -= len(sampled_items.get(str(ds_info.id), []))

        return total_differences

    def get_import_history_data(self, slug: str, limit: int = None) -> List[List]:
        """
        Get history of import tasks

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List[List]: List of import tasks with their details
        """
        import_tasks = self.al_session.state.import_tasks.get(slug, [])
        history = self.al_session.state.project.custom_data.get("import_history", {}).get(
            "tasks", []
        )
        history_dict = {item["task_id"]: item for item in history}

        data = []
        for task_id in import_tasks:
            history_item = history_dict.get(task_id)
            if history_item is None:
                data.append([task_id, "", "", "", 0, "failed"])
                continue
            if history_item.get("slug") != slug:
                logger.warning(
                    f"Import history item with task_id {task_id} does not match the slug {slug}. Skipping."
                )
                continue
            datasets = history_item.get("datasets", [])
            ds_ids = ", ".join(str(d["id"]) for d in datasets)
            status = history_item.get("status")
            if status == "started":
                status = "ok"
            row = [
                history_item.get("task_id"),
                history_item.get("app", {}).get("name", ""),
                ds_ids,
                history_item.get("timestamp"),
                history_item.get("items_count"),
                status,
            ]
            data.append(row)
            if limit is not None and len(data) >= limit:
                break

        return data

    def get_cloud_import_history_data(self) -> List[List]:
        """
        Get data for the cloud import table

        Returns:
            List[List]: List of cloud import tasks with their details
        """
        return self.get_import_history_data(CLOUD_IMPORT_SLUG)

    def get_autoimport_history_data(self) -> List[List]:
        """
        Get data for the autoimport table

        Returns:
            List[List]: List of autoimport tasks with their details
        """
        return self.get_import_history_data(AUTOIMPORT_SLUG)

    def filter_tasks_by_slug(self, import_history: list, app_slug: str):
        tasks = []
        for history_item in import_history:
            if history_item.get("slug") == app_slug:
                tasks.append(history_item.get("task_id"))
        return tasks

    def get_last_imported_images_count(self, slug: Optional[str] = None) -> Optional[int]:
        """
        Get the number of images imported in the last import task.
        This function checks the import history of the input project and returns the count of images
        imported in the last task with the specified slug.
        """
        self.al_session.state.refresh()
        input_project = self.al_session.state.project
        import_history = input_project.custom_data.get("import_history", {}).get("tasks", [])

        solutions_autoimport_tasks = self.al_session.state.import_tasks.get(slug, [])

        # update auto import tasks button
        auto_import_tasks = self.filter_tasks_by_slug(import_history, slug)
        if any([task not in solutions_autoimport_tasks for task in auto_import_tasks]):
            self.al_session.state.add_import_tasks(slug, auto_import_tasks)

        # get last imported images count
        last_imported_images_count = None
        for import_task in import_history[::-1]:
            if len(solutions_autoimport_tasks) > 0:
                if import_task.get("task_id") == solutions_autoimport_tasks[-1]:
                    last_imported_images_count = import_task.get("items_count", 0)
                    break
        return last_imported_images_count

    def get_manual_import_last_imported_images_count(self) -> Optional[int]:
        """
        Get the number of images imported in the last import task for manual import (the AutoImport app).
        """
        return self.get_last_imported_images_count(AUTOIMPORT_SLUG)
