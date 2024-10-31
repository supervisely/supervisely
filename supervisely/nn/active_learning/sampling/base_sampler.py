from abc import ABC, abstractmethod
from supervisely import Api


class BaseSampler(ABC):
    def __init__(self, api: Api, project_id: int, image_ids: list, *args, **kwargs):
        self.api = api
        self.project_id = project_id
        self.image_ids = image_ids
        # self.embedding_provider = EmbeddingProvider(project_id)

    @abstractmethod
    def sample(self, num_images: int):
        pass
    
    # def _get_all_image_ids(self):
    #     datasets = self.api.dataset.get_list(self.project_id)
    #     image_ids = []
    #     for dataset in datasets:
    #         images = self.api.image.get_list(dataset.id)
    #         image_ids.extend([image.id for image in images])
    #     return image_ids
