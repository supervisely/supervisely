from collections import defaultdict
import time
import numpy as np
from supervisely.nn.active_learning.sampling.base_sampler import BaseSampler
from supervisely import Api
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from umap import UMAP
except ImportError:
    pass
from supervisely import logger


class KMeansSampler(BaseSampler):
    def __init__(self, api: Api, project_id: int, image_ids: list, decomposition_method="UMAP", n_dim=3):
        super().__init__(api, project_id, image_ids)
        self.decomposition_method = decomposition_method
        self.n_dim = n_dim
        

    def sample(self, num_images: int):
        """
        Sample images using clustering approach with dimensionality reduction.
        
        Args:
            num_images (int): Number of images to sample
        
        Returns:
            List[str]: List of sampled image IDs
        """
        num_clusters = num_images
        image_ids = self.image_ids
        items = self.api.embeddings.get_info_by_ids(self.project_id, image_ids)
        embeddings, image_ids = zip(*[(item.vector, item.id) for item in items])
        embeddings = np.stack(embeddings, axis=0)
        
        # Dimensionality reduction
        t0 = time.time()
        if self.decomposition_method == "PCA":
            pca = PCA(n_components=self.n_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
        elif self.decomposition_method == "UMAP":
            umap = UMAP(n_components=self.n_dim)
            reduced_embeddings = umap.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown decomposition method: {self.decomposition_method}")
        
        # Clustering
        kmeans = MiniBatchKMeans(n_clusters=num_clusters)
        cluster_labels = kmeans.fit_predict(reduced_embeddings)
        
        # Associate image IDs with cluster labels
        image_clusters = defaultdict(list)
        for image_id, cluster_label in zip(image_ids, cluster_labels):
            image_clusters[cluster_label].append(image_id)
        image_clusters = dict(image_clusters)
        
        # Select representative image for each cluster
        # i.e, the image with minimum distance to cluster center
        sampled_image_ids = []
        for cluster_label in range(num_clusters):
            cluster_images = image_clusters.get(cluster_label, [])
            if not cluster_images:
                continue
            cluster_center = kmeans.cluster_centers_[cluster_label]
            cluster_embeddings = np.array([
                reduced_embeddings[image_ids.index(img_id)] 
                for img_id in cluster_images
            ])
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest_image_index = np.argmin(distances)
            sampled_image_ids.append(cluster_images[closest_image_index])
        dt = time.time() - t0
        logger.info(f"KMeans sampling completed in {dt:.2f} seconds")

        return sampled_image_ids
        