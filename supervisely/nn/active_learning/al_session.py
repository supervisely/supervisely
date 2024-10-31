from collections import defaultdict
from supervisely import Api
from supervisely.io.json import load_json_file
from supervisely.nn.active_learning.sampling.random_sampler import RandomSampler
from supervisely.nn.active_learning.sampling.kmeans_sampler import KMeansSampler
from supervisely import DatasetInfo


class ALSession:
    def __init__(self, api: Api, workspace_id: int):
        self.api = api
        self.workspace_id = workspace_id
        self.team_id = self.api.workspace.get_info_by_id(self.workspace_id).team_id
        self.index_project_id = None
        self.labeling_project_id = None
        self.training_project_id = None
        self.train_dataset_id = None
        self.val_dataset_id = None
        self._embedding_generator_id = None
        # self._checkpoints_db_connection = None
        # self._best_checkpoint_id = None
        # self._clip_settings = None
        self.not_sampled_image_ids = None  # TODO: get from DB?
        self._init_with_team_files_config()

    def sample(self, method: str, num_images: int, sampler_params: dict):
        if method == "random":
            sampler_cls = RandomSampler
        elif method == "kmeans":
            sampler_cls = KMeansSampler
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        sampler = sampler_cls(self.api, self.index_project_id, self.not_sampled_image_ids, **sampler_params)
        sampled_image_ids = sampler.sample(num_images)
        self._submit_sampling(sampled_image_ids)

    def _submit_sampling(self, sampled_image_ids: list):
        new_dataset_info = self._copy_to_labeling_project(sampled_image_ids)
        self._update_not_sampled_ids(sampled_image_ids)
    
    def _copy_to_labeling_project(self, image_ids: list) -> DatasetInfo:
        api = self.api
        existed_datasets = api.dataset.get_list(self.labeling_project_id)
        name = f"batch_{len(existed_datasets) + 1:03d}"
        new_dataset_info = api.dataset.create(self.labeling_project_id, name)
        image_infos = api.image.get_info_by_id_batch(image_ids)
        images_by_dataset = defaultdict(list)
        for image_info in image_infos:
            dataset_id = image_info.dataset_id
            images_by_dataset[dataset_id].append(image_info)
        for dataset_id, image_infos in images_by_dataset.items():
            api.image.copy_batch_optimized(dataset_id, image_infos, new_dataset_info.id)
        return new_dataset_info

    def _update_not_sampled_ids(self, sampled_image_ids: list):
        raise NotImplementedError

    def _init_with_team_files_config(self):
        local_config_path = "/tmp/config.json"
        self.api.file.download(self.team_id, f"/active_learning/{self.workspace_id}/config.json", local_config_path)
        config: dict = load_json_file(local_config_path)
        self.index_project_id = config["index_project_id"]
        self.labeling_project_id = config["labeling_project_id"]
        self.training_project_id = config["training_project_id"]
        self.train_dataset_id = config["train_dataset_id"]
        self.val_dataset_id = config["val_dataset_id"]
