from collections import defaultdict
import random
from supervisely import Api
from supervisely.io.json import load_json_file, dump_json_file
from supervisely.nn.active_learning.sampling.random_sampler import RandomSampler
from supervisely.nn.active_learning.sampling.kmeans_sampler import KMeansSampler
from supervisely.nn.active_learning.sampling.uncertainty_sampler import UncertaintySampler
from supervisely import DatasetInfo
from supervisely import logger


class ALSession:
    INDEX_PROJECT_NAME = "Index"
    LABELING_PROJECT_NAME = "Labeling"
    TRAINING_PROJECT_NAME = "Training"
    TRAIN_DATASET_NAME = "train"
    VAL_DATASET_NAME = "val"
    LABELING_BATCH_NAME = "batch"
    TRAINING_ITER_NAME = "iter"

    def __init__(
        self,
        api: Api,
        workspace_id: int,
    ):
        self.api = api
        self.workspace_id = workspace_id
        self.team_id = self.api.workspace.get_info_by_id(self.workspace_id).team_id
        self.index_project_id = None
        self.labeling_project_id = None
        self.training_project_id = None
        self.train_dataset_id = None
        self.val_dataset_id = None
        # self._embedding_generator_id = None
        self._load_from_team_files_config_if_exists()

    @staticmethod
    def from_empty_workspace(api: Api, workspace_id: int) -> "ALSession":
        api = api
        index_info = api.project.create(workspace_id, ALSession.INDEX_PROJECT_NAME)
        labeling_info = api.project.create(workspace_id, ALSession.LABELING_PROJECT_NAME)
        training_info = api.project.create(workspace_id, ALSession.TRAINING_PROJECT_NAME)
        config = {
            "index_project_id": index_info.id,
            "labeling_project_id": labeling_info.id,
            "training_project_id": training_info.id,
            "train_dataset_id": None,
            "val_dataset_id": None,
        }
        al_session = ALSession(api, workspace_id)
        al_session.load_with_config(config)
        al_session._update_team_files_config()
        return al_session

    def load_from_team_files_config(self):
        local_config_path = "/tmp/config.json"
        self.api.file.download(self.team_id, self.team_files_config_path(), local_config_path)
        config: dict = load_json_file(local_config_path)
        self.load_with_config(config)
        self._sync_config()

    def load_with_config(self, config: dict):
        self.index_project_id = config["index_project_id"]
        self.labeling_project_id = config["labeling_project_id"]
        self.training_project_id = config["training_project_id"]
        self.train_dataset_id = config["train_dataset_id"]
        self.val_dataset_id = config["val_dataset_id"]

    def team_files_config_path(self):
        return f"/active_learning/{self.workspace_id}/config.json"

    def al_config(self):
        config = {
            "index_project_id": self.index_project_id,
            "labeling_project_id": self.labeling_project_id,
            "training_project_id": self.training_project_id,
            "train_dataset_id": self.train_dataset_id,
            "val_dataset_id": self.val_dataset_id,
        }
        return config

    def sample(self, method: str, num_images: int, sampler_params: dict = {}):
        if method == "random":
            sampler_cls = RandomSampler
        elif method == "kmeans":
            sampler_cls = KMeansSampler
        elif method == "uncertainty":
            sampler_cls = UncertaintySampler
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        not_sampled_image_ids = self._not_sampled_ids()
        sampler = sampler_cls(
            self.api, self.index_project_id, not_sampled_image_ids, **sampler_params
        )
        sampled_image_ids = sampler.sample(num_images)
        new_labeling_dataset_info = self._copy_to_labeling_project(sampled_image_ids)
        return new_labeling_dataset_info

    def move_labeling_to_training(self, train_ratio: float = None, train_size: int = None):
        labeled_image_ids = self._list_images(self.labeling_project_id)
        train_ids, val_ids = self._train_val_split(labeled_image_ids, train_ratio, train_size)
        self._create_training_iteration(train_ids, val_ids)

    def _copy_to_labeling_project(self, image_ids: list) -> DatasetInfo:
        api = self.api
        if self.labeling_project_id is None:
            self.labeling_project_id = api.project.get_or_create(
                self.workspace_id, self.LABELING_PROJECT_NAME
            ).id
            self._update_team_files_config()
        existed_datasets = api.dataset.get_list(self.labeling_project_id)
        name = f"{self.LABELING_BATCH_NAME}_{len(existed_datasets) + 1:03d}"
        new_dataset_info = api.dataset.create(self.labeling_project_id, name)
        image_infos = api.image.get_info_by_id_batch(image_ids)
        images_by_dataset = defaultdict(list)
        for image_info in image_infos:
            dataset_id = image_info.dataset_id
            images_by_dataset[dataset_id].append(image_info)
        for dataset_id, image_infos in images_by_dataset.items():
            api.image.copy_batch_optimized(dataset_id, image_infos, new_dataset_info.id)
        return new_dataset_info

    def _create_training_iteration(self, train_ids: list, val_ids: list):
        # train/val split on labeled_image_ids
        # get train/val dataset ids (if exist)
        # merge new train/val datasets with existing ones
        # create iter_{i} dataset
        # create train/val nested datasets
        api = self.api
        if self.training_project_id is None:
            self.training_project_id = api.project.get_or_create(
                self.workspace_id, self.TRAINING_PROJECT_NAME
            ).id
            self._update_team_files_config()
        existed_datasets = api.dataset.get_list(self.training_project_id)
        old_train_ids, old_val_ids = self._existed_train_and_val_ids()
        train_ids.extend(old_train_ids)
        val_ids.extend(old_val_ids)
        # create new train/val datasets
        iter_num = len(existed_datasets) + 1
        iter_name = f"{self.TRAINING_ITER_NAME}_{iter_num:03d}"
        iter_dataset_info = api.dataset.create(self.training_project_id, iter_name)
        train_dataset_info = api.dataset.create(
            self.training_project_id, self.TRAIN_DATASET_NAME, parent_id=iter_dataset_info.id
        )
        val_dataset_info = api.dataset.create(
            self.training_project_id, self.VAL_DATASET_NAME, parent_id=iter_dataset_info.id
        )
        self.train_dataset_id = train_dataset_info.id
        self.val_dataset_id = val_dataset_info.id
        self._update_team_files_config()

        # copy images to train/val datasets
        def upload_images(dataset_id, image_ids):
            image_infos = api.image.get_info_by_id_batch(image_ids)
            names, hashes = zip(*[(image_info.name, image_info.hash) for image_info in image_infos])
            return api.image.upload_hashes(dataset_id, names, hashes, batch_size=100)

        upload_images(train_dataset_info.id, train_ids)
        upload_images(val_dataset_info.id, val_ids)
        # clear labeling project from labeled images
        api.image.remove_batch(train_ids + val_ids)

    def _train_val_split(
        self,
        image_ids: list,
        train_ratio: float = None,
        train_size: int = None,
        stratified: bool = False,
    ):
        # TODO: implement full method in SDK
        if train_ratio is None and train_size is None:
            raise ValueError("Either train_ratio or train_size must be specified")
        if train_ratio is not None:
            if train_ratio <= 0 or train_ratio >= 1:
                raise ValueError("train_ratio must be in (0, 1)")
            train_size = int(len(image_ids) * train_ratio)
        if train_size is not None:
            if train_size <= 0 or train_size >= len(image_ids):
                raise ValueError("train_size must be in (0, len(image_ids))")
        if stratified:
            raise NotImplementedError("Stratified split is not implemented")
        random.shuffle(image_ids)
        train_ids = image_ids[:train_size]
        val_ids = image_ids[train_size:]
        return train_ids, val_ids

    def _not_sampled_ids(self):
        # not_sampled_ids = Index - (Labeling + Training)
        index_ids = self._list_images(self.index_project_id)
        if self.labeling_project_id is not None:
            labeling_ids = self._list_images(self.labeling_project_id)
        else:
            labeling_ids = []
        if self.training_project_id is not None:
            train_ids, val_ids = self._existed_train_and_val_ids()
            training_ids = train_ids + val_ids
        else:
            training_ids = []
        not_sampled_ids = set(index_ids) - set(labeling_ids) - set(training_ids)
        return list(not_sampled_ids)

    def _update_team_files_config(self):
        config = self.al_config()
        local_config_path = "/tmp/config.json"
        dump_json_file(config, local_config_path)
        self.api.file.upload(self.team_id, local_config_path, self.team_files_config_path())

    def _list_images(self, project_id: int):
        datasets = self.api.dataset.get_list(project_id)
        image_ids = []
        for dataset in datasets:
            images = self.api.image.get_list(dataset.id)
            image_ids.extend([image.id for image in images])
        return image_ids

    def _existed_train_and_val_ids(self):
        train_ids = []
        val_ids = []
        if self.train_dataset_id is not None:
            train_ids = self.api.image.get_list(self.train_dataset_id)
            train_ids = [image.id for image in train_ids]
        if self.val_dataset_id is not None:
            val_ids = self.api.image.get_list(self.val_dataset_id)
            val_ids = [image.id for image in val_ids]
        return train_ids, val_ids

    def _load_from_team_files_config_if_exists(self):
        if self.api.file.exists(self.team_id, self.team_files_config_path()):
            self.load_from_team_files_config()
            logger.debug("AL session loaded from team files config")
            return True
        else:
            logger.debug("Team files config not found")
            return False

    def _sync_config(self):
        # check if projects and datasets still exist
        # if not, set corresponding fields to None
        if not self.api.project.get_info_by_id(self.index_project_id):
            self.index_project_id = None
        if not self.api.project.get_info_by_id(self.labeling_project_id):
            self.labeling_project_id = None
        if not self.api.project.get_info_by_id(self.training_project_id):
            self.training_project_id = None
        if self.training_project_id is not None:
            if self.train_dataset_id and not self.api.dataset.get_info_by_id(self.train_dataset_id):
                self.train_dataset_id = None
            if self.val_dataset_id and not self.api.dataset.get_info_by_id(self.val_dataset_id):
                self.val_dataset_id = None
        self._update_team_files_config()
