from typing import List, NamedTuple

import numpy as np

# from supervisely.api.api import Api

# TODO: Replace with relative endpoint.
# E.g. <instance-address>/embeddings
QDRANT_ENDPOINT = "https://qdrant-dev.internal.supervisely.com:443"

try:
    from qdrant_client import QdrantClient
except ImportError:
    pass


class EmbeddingInfo(NamedTuple):
    """Represents the information about the embedding, which contains information about
    the image in Supervisely and the vector in the embedding space.
    """

    id: int
    vector: np.ndarray


class EmbeddingsApi:
    def __init__(self, api):
        self._api = api

    def _get_client(self, host: str = None) -> QdrantClient:
        if host is None:
            # TODO: Enable this later.
            # host = f"{self._api.server_address}/{QDRANT_ENDPOINT}"
            host = QDRANT_ENDPOINT
            # For now, we are using the development server.
        try:
            return QdrantClient(host)
        except NameError:
            raise ImportError(
                "QdrantClient is not installed. "
                "Please install it using `pip install qdrant-client`."
            )

    def get_info_by_id(self, project_id: int, image_id: int):
        return self.get_info_by_ids(project_id, [image_id])[0]

    def get_info_by_ids(self, project_id: int, image_ids: List[int]) -> List[EmbeddingInfo]:
        client = self._get_client()

        # If any additional fields will be added to the database (e.g. dataset_id, image_name, etc.),
        # the payload can be used to retrieve them.
        # In this case the EmbeddingInfo should be extended with the corresponding fields.

        points = client.retrieve(
            str(project_id), image_ids, with_vectors=True
        )  # , with_payload=True)

        return [EmbeddingInfo(id=point.id, vector=np.array(point.vector)) for point in points]
