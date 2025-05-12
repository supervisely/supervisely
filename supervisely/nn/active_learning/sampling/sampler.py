from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from supervisely.api.dataset_api import DatasetInfo
from supervisely.nn.active_learning.sampling.constants import (
    SamplingMode,
    SamplingSettings,
)
from supervisely.nn.active_learning.scheduler.scheduler import SchedulerJobs
from supervisely.nn.active_learning.utils.constants import DATA_ORGANIZER_SLUG

if TYPE_CHECKING:
    from supervisely.nn.active_learning.session import ActiveLearningSession

import random

from supervisely.api.image_api import ImageInfo
from supervisely.nn.active_learning.utils.constants import EMBEDDINGS_GENERATOR_SLUG
from supervisely.nn.active_learning.utils.project import get_diffs
from supervisely.sly_logger import logger


class ActiveLearningSampler:
    def __init__(self, al_session):
        self.al_session: ActiveLearningSession = al_session
        self.api = al_session.api
        self.project_id = al_session.project_id
        self.workspace_id = al_session.workspace_id
        self.team_id = al_session.team_id
        self.state = al_session.state

    # old implementation (using redundant API calls)
    # def wait_sampling_app_is_ready(self, task_id: int) -> bool:
    #     """Wait for sampling app to be ready for API calls"""
    #     try:
    #         self.api.app.wait(task_id, target_status=self.api.task.Status.STARTED)
    #         return True
    #     except Exception as e:
    #         logger.error(f"Sampling task {task_id} failed: {str(e)}")
    #         return False

    # def wait_sampling_completion(self, task_id: int) -> bool:
    #     """Wait for sampling task to complete and return status"""
    #     try:
    #         self.api.app.wait(task_id, target_status=self.api.task.Status.FINISHED)
    #         return True
    #     except Exception as e:
    #         logger.error(f"Import task {task_id} failed: {str(e)}")
    #         return False

    # def run_sampling(self, sampling_settings: SamplingSettings) -> int:
    #     task_id = self.run_sampling_app()
    #     self.wait_sampling_app_is_ready(task_id)
    #     res = self.send_sampling_request(task_id, sampling_settings)
    #     return res

    # def run_sampling_app(self) -> int:
    #     """Run sampling app task"""
    #     module_info = self.api.app.get_ecosystem_module_info(slug=DATA_ORGANIZER_SLUG)

    #     session = self.api.app.start(
    #         agent_id=49,
    #         module_id=module_info.id,
    #         workspace_id=self.al_session.workspace_id,
    #         task_name="Sample data",
    #         params={},
    #     )
    #     self.al_session.state.add_sampling_task(task_id=session.task_id)
    #     # self.api.app.wait(session.task_id, target_status=self.api.task.Status.STARTED)
    #     # self.al_session.scheduler.add_job(
    #     #     SchedulerJobs.SEND_SAMPLING_REQUEST,
    #     #     self.send_sampling_request,
    #     #     10,
    #     #     args=[session.task_id, sampling_settings],
    #     # )
    #     logger.info(f"Started sampling task: {session.task_id}")
    #     return session.task_id

    # def send_sampling_request(
    #     self, task_id: int, sampling_settings: SamplingSettings
    # ) -> Union[bool, None]:
    #     """Send sampling request via API to Sampling App session."""
    #     ready = self.api.app.is_ready_for_api_calls(task_id)
    #     if not ready:
    #         logger.info(f"Session is not ready for API calls. Waiting...")
    #         return
    #     self.al_session.scheduler.remove_job(SchedulerJobs.SEND_SAMPLING_REQUEST)
    #     logger.info(f"Sampling session is ready for API calls: {task_id}")

    #     sampled_images = self.al_session.state.get_sampled_images()
    #     sampling_mode = sampling_settings.get("mode", "Random").lower()

    #     data = {
    #         "src_project_id": self.project_id,
    #         "dst_project_id": self.al_session.state.labeling_project_id,
    #         "mode": sampling_mode,
    #         "sampled_images": sampled_images,
    #         "sample_size": sampling_settings.get("sample_size", None),
    #         "diversity_mode": sampling_settings.get("diversity_mode", None),
    #         "prompt": sampling_settings.get("prompt", None),
    #     }
    #     res = self.api.app.send_request(task_id, "sample", data)
    #     all_img_ids = []
    #     if res is not None:
    #         if "error" in res:
    #             logger.error(f"Error: {res['error']}")
    #         elif "data" in res:
    #             data = res["data"]
    #             src = data.get("src", {})
    #             dst = data.get("dst", {})
    #             if src and dst:
    #                 for dst_imgs in dst.values():
    #                     all_img_ids.extend(dst_imgs)
    #                 self.al_session.state.add_sampling_batch(batch_data=src)
    #                 labeling_collection_id = self.al_session.state.labeling_collection_id
    #                 self.api.entities_collection.add_items(labeling_collection_id, all_img_ids)
    #                 logger.info(f"Copied {len(all_img_ids)} images to labeling project")

    #     # stop app
    #     self.api.app.stop(task_id)
    #     return len(all_img_ids)

    def schedule_sampling(self, settings: SamplingSettings, interval: int) -> str:
        """
        Schedule a sampling task

        Args:

            interval (int): Interval in seconds for the scheduled task.
        """
        return self.al_session.scheduler.add_job(
            job_id=SchedulerJobs.START_SAMPLING,
            func=self.sample,
            interval_sec=interval,
            args=(settings,),
        )

    def unschedule_sampling(self) -> bool:
        """
        Unschedule the sampling task
        """
        return self.al_session.scheduler.remove_job(SchedulerJobs.START_SAMPLING)

    def get_sampling_history_data(self) -> List[List]:
        """
        Get data for the sampling history table

        Returns:
            List[List]: List of sampling tasks with their details
        """
        # sampling_tasks = self.al_session.state.get_sampling_tasks()
        sampling_history = self.al_session.state.project.custom_data.get(
            "sampling_history", {}
        ).get("tasks", [])
        # history_dict = {item["task_id"]: item for item in sampling_history}

        rows = []
        for history_item in sampling_history:
            # history_item = history_dict.get(task_id)
            # if history_item is None:
            #     rows.append([task_id, "", "", "", 0, "failed"])
            #     continue
            row = [
                history_item.get("mode"),
                history_item.get("status"),
                history_item.get("timestamp"),
                history_item.get("items_count"),
            ]
            rows.append(row)
        return rows

    def sample(self, settings: SamplingSettings) -> None:
        """Sample data using the specified sampling settings."""

        # Create a mapping with different between source and destination datasets
        diffs = get_diffs(self.api, self.project_id, self.al_session.state.labeling_project_id)
        diff_images, src_to_dst_map, ds_to_create = diffs

        # If there is no difference between the datasets, return None
        if not diff_images:
            logger.warning("No new items to copy to the labeling project")
            self._add_record_to_history(
                status="error", total_items=0, items=[], mode=settings["mode"]
            )
            return {"src": None, "dst": None}

        # Prepare the sample
        new_sampled_images = self._prepare_sample(diff_images, settings)
        if new_sampled_images is None:
            logger.warning("No new items to copy to the labeling project")
            self._add_record_to_history(
                status="error", total_items=0, items=[], mode=settings["mode"]
            )
            return {"src": None, "dst": None}

        # # Copy the sampled images to the destination project
        src, added = self._copy_to_labeling_project(
            src_to_dst_map, new_sampled_images, ds_to_create
        )

        all_img_ids = []
        for dst_imgs in added.values():
            all_img_ids.extend(dst_imgs)
        self.al_session.state.add_sampling_batch(batch_data=src)
        labeling_collection_id = self.al_session.state.labeling_collection_id
        self.api.entities_collection.add_items(labeling_collection_id, all_img_ids)
        logger.info(f"Copied {len(all_img_ids)} images to labeling project")

        res = {"src": src, "dst": added}
        # Add record to history
        self._add_record_to_history(
            status="completed", total_items=len(all_img_ids), items=res, mode=settings["mode"]
        )

        return len(all_img_ids)

    def preview(self, settings: SamplingSettings, limit: int) -> None:
        """Preview the sampling images without copying them to the labeling project."""

        # Create a mapping with different between source and destination datasets
        diffs = get_diffs(self.api, self.project_id, self.al_session.state.labeling_project_id)
        diff_images, _, _ = diffs

        # If there is no difference between the datasets, return None
        if not diff_images:
            logger.warning("No new items to preview")
            return []

        # Prepare the sample
        new_sampled_images = self._prepare_sample(diff_images, settings)
        if new_sampled_images is None:
            logger.warning("Failed to prepare sample for preview")
            return []

        # flatten the list of images and limit the number of images
        all_sampled_images = []
        for imgs in new_sampled_images.values():
            all_sampled_images.extend(imgs)
        all_sampled_images = all_sampled_images[:limit]

        image_urls = [img.full_storage_url for img in all_sampled_images]
        return image_urls

    def _prepare_sample(
        self,
        diffs: Dict[int, List[ImageInfo]],
        settings: Dict[str, Union[int, str]],
    ) -> Dict[int, List[ImageInfo]]:
        """
        Prepare a sample of images from the differences and sample size.
        Args:
            diffs (dict): Dictionary of differences between source and destination datasets.
            settings (dict): Dictionary with sampling settings:
                - mode (str): Mode of sampling. Can be "random", "diverse" or "ai search".
                - sample_size (int): Number of images to sample. Optional.
                - diversity_mode (str): Mode of diversity. Can be "random" or "diverse". Optional.
                - prompt (str): Prompt for AI search. Optional.
        Returns:
            dict: Dictionary of sampled images.
        """
        sampled_images = self.al_session.state.get_sampled_images()
        mode = settings["mode"]
        # Filter out already sampled images
        diffs = self._filter_diffs(diffs, sampled_images)
        # Calculate the total number of differences
        total_diffs = sum(len(imgs) for imgs in diffs.values())

        # If there are no differences
        if total_diffs == 0:
            logger.warning("No new items to copy to the labeling project")
            return None

        # If the sample size is greater than the total differences, return all images
        sample_size = settings.get("sample_size", None)
        if sample_size and sample_size >= total_diffs:
            logger.warning(
                f"Sample size ({sample_size}) is greater than total differences ({total_diffs}). "
                "Returning all images."
            )
            return diffs

        # Calculate the sample size for each dataset
        samples_per_dataset = {}
        remaining = sample_size or total_diffs
        for ds_id, imgs in diffs.items():
            # Calculate proportional size and round down
            if sample_size is not None:
                ds_sample = int((len(imgs) / total_diffs) * sample_size)
            else:
                ds_sample = len(imgs)
            samples_per_dataset[ds_id] = ds_sample
            remaining -= ds_sample

        # Distribute any remaining samples randomly
        if remaining > 0:
            datasets_with_space = [
                ds_id for ds_id, imgs in diffs.items() if len(imgs) > samples_per_dataset[ds_id]
            ]
            while remaining > 0 and datasets_with_space:
                ds_id = random.choice(datasets_with_space)
                if len(diffs[ds_id]) > samples_per_dataset[ds_id]:
                    samples_per_dataset[ds_id] += 1
                    remaining -= 1
                else:
                    datasets_with_space.remove(ds_id)

        if mode == SamplingMode.RANDOM.value:
            new_sampled_images = {}
            for ds_id, sample_count in samples_per_dataset.items():
                if sample_count > 0:
                    new_sampled_images[ds_id] = random.sample(diffs[ds_id], sample_count)
        elif mode in [SamplingMode.DIVERSE.value, SamplingMode.AI_SEARCH.value]:
            all_diffs_flat = []
            for ds_id, imgs in diffs.items():
                all_diffs_flat.extend([img.id for img in imgs])
            logger.info(f"Sample mode: {mode}. Settings: {settings}")

            method = "diverse" if mode == SamplingMode.DIVERSE.value else "search"
            data = {"project_id": self.project_id}
            data["image_ids"] = all_diffs_flat

            if mode == SamplingMode.AI_SEARCH.value:
                # AI search mode
                prompt = settings.get("prompt", None)
                if prompt is None:
                    logger.error("Prompt is required for AI search mode.")
                    return None
                data["prompt"] = prompt
                data["limit"] = settings.get("limit", None)
            elif mode == SamplingMode.DIVERSE.value:
                # Diverse mode
                data["sample_size"] = sample_size
                data["method"] = settings.get("diversity_mode", "centroids")
            else:
                logger.error(f"Unknown sampling mode: {mode}")
                return None
            # Send request to the API
            module_info = self.api.app.get_ecosystem_module_info(slug=EMBEDDINGS_GENERATOR_SLUG)
            sessions = self.api.app.get_sessions(
                self.team_id, module_info.id, statuses=[self.api.task.Status.STARTED]
            )
            if len(sessions) == 0:
                logger.error("No active sessions found for embeddings generator.")
                return None
            session = sessions[0]
            # api.app.wait(session.task_id, target_status=api.task.Status.STARTED)
            logger.info(f"Embeddings generator session: {session.task_id}")
            res = self.api.app.send_request(session.task_id, method, data=data)
            if isinstance(res, dict):
                if "collection_id" in res:
                    collection_id = res["collection_id"]
                    all_sampled_images = self.api.entities_collection.get_items(collection_id)
                    all_sampled_ids = [img.id for img in all_sampled_images]
                    new_sampled_images = {
                        ds_id: [img for img in diffs[ds_id] if img.id in all_sampled_ids]
                        for ds_id in diffs.keys()
                    }
                elif "message" in res:
                    logger.error(f"Error during sampling: {res['message']}")
                    return None
            elif isinstance(res, list):
                res_ids = {img["id"] for img in res}
                new_sampled_images = {
                    ds_id: [img for img in diffs[ds_id] if img.id in res_ids]
                    for ds_id in diffs.keys()
                }
            else:
                logger.error(f"Error during sampling: {res}")
                return None

        else:
            logger.error(f"Unknown sampling mode: {mode}")
            return None

        return new_sampled_images

    def _filter_diffs(
        self,
        all_diffs: Dict[int, List[ImageInfo]],
        sampled_images: Dict[int, List[int]],
    ) -> Dict[int, List[ImageInfo]]:
        """
        Filter out images that have already been sampled.

        Args:
            all_diffs: Dictionary of all differences between source and destination datasets.
            sampled_images: Dictionary of already sampled images.

        Returns:
            Dictionary of filtered differences.
        """
        filtered_diffs = {}
        for ds_id, imgs in all_diffs.items():
            ignore_ids = {img for img in sampled_images.get(ds_id, [])}
            filtered_diffs[ds_id] = [img for img in imgs if img.id not in ignore_ids]
        return filtered_diffs

    def _copy_to_labeling_project(
        self,
        src_to_dst_map: Dict[int, int],
        sampled_images: Dict[int, List[ImageInfo]],
        ds_to_create: List[int],
        remove_src: bool = False,
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Copy or move images from the source project to the destination project.

        Args:
            src_to_dst_map (dict): Mapping of source dataset IDs to destination dataset IDs.
            sampled_images (dict): Dictionary of sampled images.
            ds_to_create (list): List of datasets to create in the destination project.
            remove_src (bool): Whether to remove the source images after copying.
        Returns:
            dict: Dictionary with source and destination dataset IDs.
        """
        # Prepare children-parent relationships for source and destination datasets
        src_datasets = self.api.dataset.get_list(self.project_id, recursive=True)
        src_id_to_info = {ds.id: ds for ds in src_datasets}
        src_child_to_parents = {ds.id: [] for ds in src_datasets}
        for ds in src_datasets:
            current = ds
            while parent_id := current.parent_id:
                src_child_to_parents[ds.id].append(parent_id)
                current = src_id_to_info[parent_id]

        added = {}
        src = {}
        for src_ds_id, src_imgs in sampled_images.items():
            if len(src_imgs) > 0:
                dst_ds_id = src_to_dst_map.get(src_ds_id)
                if dst_ds_id is None and src_ds_id in ds_to_create:
                    # Create new dataset in destination project
                    src_parent_ids = src_child_to_parents[src_ds_id]
                    dst_parent_id = None
                    for parent_id in src_parent_ids:
                        src_ds = self.api.dataset.get_info_by_id(parent_id)
                        dst_ds = self.api.dataset.create(
                            self.al_session.state.labeling_project_id,
                            src_ds.name,
                            parent_id=dst_parent_id,
                        )
                        dst_parent_id = dst_ds.id
                        src_to_dst_map[parent_id] = dst_parent_id

                    # Create new dataset in destination project
                    src_ds = self.api.dataset.get_info_by_id(src_ds_id)
                    dst_ds = self.api.dataset.create(
                        self.al_session.state.labeling_project_id,
                        src_ds.name,
                        parent_id=dst_parent_id,
                    )
                    dst_ds_id = dst_ds.id
                    src_to_dst_map[src_ds_id] = dst_ds_id

                new_imgs = self.api.image.copy_batch_optimized(
                    src_dataset_id=src_ds_id,
                    src_image_infos=src_imgs,
                    dst_dataset_id=dst_ds_id,
                    with_annotations=True,
                    save_source_date=False,
                )

                if remove_src:
                    # Remove source images if requested
                    self.api.image.remove_batch([i.id for i in src_imgs], batch_size=200)

                src[src_ds_id] = [i.id for i in src_imgs]
                added[dst_ds_id] = [i.id for i in new_imgs]
                logger.info(f"Copied {len(new_imgs)} images to dataset {dst_ds_id}")
        return src, added

    def _add_record_to_history(
        self,
        status: Literal["completed", "error"],
        total_items: int = None,
        items: List[int] = None,
        mode: Optional[Literal[""]] = None,
    ) -> None:
        """
        Adds a record to the Project's sample or move history.
        Args:
            status (str): Status of the task ("completed" or "failed").
            total_items (int): Total number of items in the task.
            items (list): List of item IDs.
        """
        key = "sampling_history"

        data = {
            "status": status,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "items_count": total_items,
            "items": items,
            "key": key,
        }
        if mode is not None:
            data["mode"] = mode

        project_info = self.api.project.get_info_by_id(self.project_id)

        custom_data = project_info.custom_data or {}
        if key not in custom_data:
            custom_data[key] = {"tasks": []}
        if "tasks" not in custom_data[key]:
            custom_data[key]["tasks"] = []
        custom_data[key]["tasks"].append(data)

        self.api.project.edit_info(self.project_id, custom_data=custom_data)
