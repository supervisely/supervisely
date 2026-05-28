from typing import Dict, List, Literal, Tuple, Union

from supervisely.nn.active_learning.state.managers.project_state_manager import (
    StateManager,
)


class TrainValSplitStateManager:
    """Handles dataset splitting"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def get_split_collections(self) -> Dict[str, List[int]]:
        """Get split collections"""
        return self.state_manager.get("splits_collections", {"train": [], "val": []})

    def add_split_collections(self, key: str, collection_ids: List[int]) -> None:
        """Add split collection IDs to state"""
        collections = self.get_split_collections()

        if key not in collections:
            collections[key] = []

        for collection_id in collection_ids:
            if collection_id not in collections[key]:
                collections[key].append(collection_id)

        self.state_manager.set("splits_collections", collections)

    def get_split_settings(self) -> Tuple[str, Union[float, List[str]], Union[float, List[str]]]:
        """Get split settings"""
        settings = self.state_manager.get("split_settings", {})
        mode = settings.get("mode", "random")
        train = settings.get("train", 0.8)
        val = settings.get("val", 0.2)

        return mode, train, val

    def set_split_settings(
        self,
        mode: Literal["random", "datasets"],
        train: Union[float, List[str]],
        val: Union[float, List[str]],
    ) -> None:
        """Set split settings"""
        if mode not in ("random", "datasets"):
            raise ALStateError(f"Split mode {mode} is not supported")

        if isinstance(train, int) and isinstance(val, int):
            total = train + val
            train = round(train / total, 2)
            val = round(val / total, 2)

        if isinstance(train, float) and isinstance(val, float):
            if not 0.99 <= (train + val) <= 1.01:  # Allow small rounding errors
                raise ValueError(f"Train ({train}) and validation ({val}) splits must sum to 1.0")

        settings = {"mode": mode, "train": train, "val": val}
        self.state_manager.set("split_settings", settings)
