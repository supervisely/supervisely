import random
from supervisely.nn.active_learning.sampling.base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def sample(self, num_images: int):
        image_ids = self._get_all_image_ids()
        sampled_image_ids = random.sample(image_ids, num_images)
        return sampled_image_ids
    