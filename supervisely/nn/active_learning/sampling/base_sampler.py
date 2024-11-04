from abc import ABC, abstractmethod
from supervisely import Api


class BaseSampler(ABC):
    def __init__(self, api: Api, project_id: int, image_ids: list, *args, **kwargs):
        """
        Args:
            api (sly.Api): Supervisely API instance
            project_id (int): Project ID to sample images from
            image_ids (list): List of image IDs to sample from
        """
        self.api = api
        self.project_id = project_id
        self.image_ids = image_ids

    @abstractmethod
    def sample(self, num_images: int):
        pass
    