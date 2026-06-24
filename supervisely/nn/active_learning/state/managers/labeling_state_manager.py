from typing import List, Optional

from supervisely.api.api import Api
from supervisely.nn.active_learning.state.managers.project_state_manager import (
    StateManager,
)


class LabelingStateManager:
    """Handles labeling tasks and queues"""

    def __init__(self, state_manager: StateManager, api: Api):
        self.state_manager = state_manager
        self.api = api

    @property
    def project_id(self) -> Optional[int]:
        """Get labeling project ID"""
        return self.state_manager.get("labeling_project_id")

    def set_labeling_project_id(self, project_id: int) -> None:
        """Set labeling project ID"""
        self.state_manager.set("labeling_project_id", project_id)

    @property
    def queue_id(self) -> Optional[int]:
        """Get labeling queue ID"""
        return self.state_manager.get("labeling_queue_id")

    def set_labeling_queue_id(self, queue_id: int) -> None:
        """Set labeling queue ID"""
        self.state_manager.set("labeling_queue_id", queue_id)

    @property
    def collection_id(self) -> Optional[int]:
        """Get labeling collection ID"""
        return self.state_manager.get("labeling_collection_id")

    def set_labeling_collection_id(self, collection_id: int) -> None:
        """Set labeling collection ID"""
        self.state_manager.set("labeling_collection_id", collection_id)

    def get_new_labeled_images(self) -> List[int]:
        """Get all labeled images from labeling queue with status accepted"""
        queue_id = self.queue_id
        collection_id = self.collection_id

        if not queue_id or not collection_id:
            return []

        resp = self.api.labeling_queue.get_entities_all_pages(
            queue_id,
            collection_id,
            status="accepted",
            filter_by=None,
        )

        return resp["images"]
