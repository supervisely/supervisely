import random
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from supervisely.api.entities_collection_api import CollectionItem
from supervisely.api.image_api import ImageInfo
from supervisely.nn.active_learning.scheduler.scheduler import SchedulerJobs

if TYPE_CHECKING:
    from supervisely.nn.active_learning.session import ActiveLearningSession

from supervisely.api.api import Api
from supervisely.labeling_jobs.utils import Status
from supervisely.nn.active_learning.utils.project import (
    create_dataset_mapping,
    merge_update_metas,
)
from supervisely.sly_logger import logger


class LabelingService:

    def __init__(self, al_session):
        self.al_session: ActiveLearningSession = al_session
        self.api: Api = al_session.api
        self.project_id = al_session.project_id
        self.workspace_id = al_session.workspace_id
        self.team_id = al_session.team_id
        self.state = al_session.state

    def get_labeling_stats(self):
        logger.info("Checking labeling queue info...")
        queue_id = self.al_session.state.labeling_queue_id

        pending, annotating, reviewing, rejected, finished = 0, 0, 0, 0, 0
        queue_info = self.api.labeling_queue.get_info_by_id(queue_id)
        jobs = [self.api.labeling_job.get_info_by_id(job_id) for job_id in queue_info.jobs]
        completed = queue_info.status == Status.COMPLETED
        completed = completed or all(j.status == Status.COMPLETED for j in jobs)
        if completed:
            logger.info(f"Labeling queue is completed:: {queue_id}.")
            members = self.api.user.get_team_members(self.team_id)
            user_ids = [user.id for user in members]
            name = "Collection for Solutions"
            queue_info = self.api.labeling_queue.create(
                name=name,
                user_ids=user_ids,
                reviewer_ids=user_ids,
                collection_id=self.al_session.state.labeling_collection_id,
                allow_review_own_annotations=True,
                skip_complete_job_on_empty=True,
            )
            self.al_session.state.labeling_queue_id = queue_info.id
            logger.info(f"Created new labeling queue: {queue_info.id}")
            jobs = [self.api.labeling_job.get_info_by_id(job_id) for job_id in queue_info.jobs]

        finished += queue_info.accepted_count
        reviewing = self.api.labeling_queue.get_entities_count_by_status(queue_id, "done")
        annotating = queue_info.in_progress_count
        pending += queue_info.pending_count
        reviewing = queue_info.entities_count - annotating - pending - finished
        for job in jobs:
            for entity in job.entities:
                if entity["reviewStatus"] == "rejected":
                    rejected += 1

        logger.info(
            f"Labeling queue info: {queue_id}:\n"
            f"Pending: {pending}\n"
            f"Annotating: {annotating}\n"
            f"Reviewing: {reviewing}\n"
            f"Finished: {finished}\n"
            f"Rejected: {rejected}"
        )

        return {
            "pending": pending,
            "annotating": annotating,
            "reviewing": reviewing,
            "finished": finished,
            "rejected": rejected,
        }

    def move_to_training_project(self, min_batch: Optional[int] = None):
        new_imgs = self.al_session.state.get_new_labeled_images()
        if len(new_imgs) == 0:
            logger.info("No new labeled images found")
            return
        if min_batch is not None and len(new_imgs) < min_batch:
            logger.info(f"Not enough images to move: {len(new_imgs)} < min_batch ({min_batch})")
            return

        image_ids = [i["id"] for i in new_imgs]
        logger.info("Copying data to training project...")

        split_settings = self.al_session.state.get_split_settings()
        src_project_id = self.al_session.state.labeling_project_id
        dst_project_id = self.al_session.state.training_project_id

        num_added = 0
        preview_urls, counts = None, None
        train_collection, val_collection = None, None

        # Get source and destination projects from the API

        src_ds_tree = self.api.dataset.get_tree(src_project_id)
        dst_ds_tree = self.api.dataset.get_tree(dst_project_id)

        # Create a mapping with different between source and destination datasets
        src_to_dst_map, ds_to_create = create_dataset_mapping(src_ds_tree, dst_ds_tree)

        merge_update_metas(self.api, src_project_id, dst_project_id)

        img_infos = self.api.image.get_info_by_id_batch(image_ids, force_metadata_for_links=False)
        items = defaultdict(list)
        for img_info in img_infos:
            items[img_info.dataset_id].append(img_info)

        # Move the images to the destination project
        _, added = self._move_to_images(src_to_dst_map, items, ds_to_create)
        num_added = sum([len(i) for i in added.values()])
        logger.info(f"Copied {num_added} images to the destination project")

        # Remove the images from the source collection
        self.api.entities_collection.remove_items(
            self.al_session.state.labeling_collection_id, image_ids
        )

        # add the images to the destination collections (train/val)
        if split_settings is not None:
            mode, train, val = split_settings
            logger.info(f"Splitting images into train and val sets: {mode}")
            if mode == "random":
                train_count = int(num_added * train)
                val_count = num_added - train_count
                new_img_ids = []
                for img_ids in added.values():
                    new_img_ids.extend(img_ids)
                random.shuffle(new_img_ids)
                train_ids = new_img_ids[:train_count]
                val_ids = new_img_ids[train_count : train_count + val_count]
                logger.info(
                    f"Splitting {num_added} images into train ({len(train_ids)}) and val ({len(val_ids)})"
                )

                if len(train_ids) > 0 and len(val_ids) > 0:
                    existing_t, existing_v, split_idx = self._get_splits_details(dst_project_id)
                    train_name = f"train_{split_idx + 1}"
                    val_name = f"val_{split_idx + 1}"
                    train_collection = self.api.entities_collection.create(
                        dst_project_id, train_name
                    )
                    val_collection = self.api.entities_collection.create(dst_project_id, val_name)
                    logger.info(
                        f"Created collections '{train_collection.name}' and '{val_collection.name}' for new train and val sets"
                    )

                    train_items = [CollectionItem(entity_id=i) for i in train_ids]
                    val_items = [CollectionItem(entity_id=i) for i in val_ids]
                    self.api.entities_collection.add_items(train_collection.id, train_items)
                    self.api.entities_collection.add_items(val_collection.id, val_items)

                    logger.info(
                        f"Added {len(train_ids)} images to train collection '{train_collection.name}' "
                        f"and {len(val_ids)} images to val collection '{val_collection.name}'"
                    )
                    random_train = random.choice(train_ids)
                    random_val = random.choice(val_ids)

                    counts = [existing_t + len(train_ids), existing_v + len(val_ids)]
                    preview_urls = [
                        self.api.image.get_info_by_id(random_train).preview_url,
                        self.api.image.get_info_by_id(random_val).preview_url,
                    ]

        self.al_session.state.add_split_collection("train", train_collection)
        self.al_session.state.add_split_collection("val", val_collection)
        return {"num_added": num_added, "preview_urls": preview_urls, "counts": counts}

    def _get_splits_details(self, project_id: int):
        all_collections = self.api.entities_collection.get_list(project_id)
        train_collections = []
        val_collections = []
        for collection in all_collections:
            collection
            if collection.name.startswith("train_"):
                train_collections.append(collection)
            elif collection.name.startswith("val_"):
                val_collections.append(collection)

        if len(train_collections) != len(val_collections):
            logger.warning(
                f"Number of train collections ({len(train_collections)}) "
                f"does not match number of val collections ({len(val_collections)})"
            )

        train_count = 0
        val_count = 0
        for collection in train_collections:
            train_count += len(self.api.entities_collection.get_items(collection.id, project_id))

        for collection in val_collections:
            val_count += len(self.api.entities_collection.get_items(collection.id, project_id))

        print(f"train_count: {train_count}, val_count: {val_count}")
        return train_count, val_count, len(train_collections)

    def _move_to_images(
        self,
        src_to_dst_map: Dict[int, int],
        sampled_images: Dict[int, List[ImageInfo]],
        ds_to_create: List[int],
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Move images from the labeling project to the training project.

        Args:
            src_to_dst_map (dict): Mapping of source dataset IDs to destination dataset IDs.
            sampled_images (dict): Dictionary of sampled images.
            ds_to_create (list): List of datasets to create in the destination project.
            remove_src (bool): Whether to remove the source images after copying.
        Returns:
            dict: Dictionary with source and destination dataset IDs.
        """
        # Prepare children-parent relationships for source and destination datasets
        src_project_id = self.al_session.state.labeling_project_id
        src_datasets = self.api.dataset.get_list(src_project_id, recursive=True)
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
                            self.al_session.state.training_project_id,
                            src_ds.name,
                            parent_id=dst_parent_id,
                        )
                        dst_parent_id = dst_ds.id
                        src_to_dst_map[parent_id] = dst_parent_id

                    # Create new dataset in destination project
                    src_ds = self.api.dataset.get_info_by_id(src_ds_id)
                    dst_ds = self.api.dataset.create(
                        self.al_session.state.training_project_id,
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

                src[src_ds_id] = [i.id for i in src_imgs]
                added[dst_ds_id] = [i.id for i in new_imgs]
                logger.info(f"Copied {len(new_imgs)} images to dataset {dst_ds_id}")
        return src, added

    def schedule_refresh(self, func, interval: int = 20) -> None:
        """
        Schedule a job to refresh labeling information at a specified interval.
        """
        if interval <= 0:
            raise ValueError("Interval must be greater than 0 seconds")
        self.al_session.scheduler.add_job(SchedulerJobs.LABELING_QUEUE_STATS, func, interval)

    def schedule_move_to_training_project(
        self, func, interval: int = 20, min_batch: Optional[int] = None
    ) -> None:
        """
        Schedule a job to move labeled images to the training project at a specified interval.
        """
        if interval <= 0:
            raise ValueError("Interval must be greater than 0 seconds")
        self.al_session.scheduler.add_job(
            SchedulerJobs.MOVE_TO_TRAINING,
            func,
            interval,
            args=(min_batch,),
        )

    def unschedule_move_to_training_project(self) -> None:
        """
        Unschedule the job to move labeled images to the training project.
        """
        self.al_session.scheduler.remove_job(SchedulerJobs.MOVE_TO_TRAINING)

    def add_annotators(self, annotators: List[int]) -> None:
        """
        Add annotators to the labeling queue.
        """
        queue_id = self.al_session.state.labeling_queue_id
        self.api.labeling_queue.add_annotators(queue_id, annotators)

    def add_reviewers(self, reviewers: List[int]) -> None:
        """
        Add reviewers to the labeling queue.
        """
        queue_id = self.al_session.state.labeling_queue_id
        self.api.labeling_queue.add_reviewers(queue_id, reviewers)
