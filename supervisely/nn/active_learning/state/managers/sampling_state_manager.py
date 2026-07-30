from typing import Any, Dict, List

from supervisely.nn.active_learning.sampling.constants import SamplingMode
from supervisely.nn.active_learning.state.managers.project_state_manager import (
    StateManager,
)
from supervisely.sly_logger import logger


class SamplingStateManager:
    """Handles sampling tasks and settings"""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def get_sampling_tasks(self) -> List[int]:
        """Get all sampling tasks"""
        return self.state_manager.get("sampling_tasks", [])

    def add_sampling_tasks(self, task_ids: List[int]) -> None:
        """Add sampling task IDs to state"""
        tasks = self.get_sampling_tasks()

        for task_id in task_ids:
            if task_id not in tasks:
                tasks.append(task_id)

        self.state_manager.set("sampling_tasks", tasks)

    def get_sampling_settings(self) -> Dict[str, Any]:
        """Get sampling settings"""
        return self.state_manager.get("sampling_settings", {})

    def get_sampling_mode(self) -> SamplingMode:
        """Get current sampling mode"""
        settings = self.get_sampling_settings()
        mode = settings.get("mode", SamplingMode.RANDOM.value)
        return SamplingMode(mode).value

    def save_sampling_settings(self, settings: dict) -> None:
        """Save sampling settings with validation"""

        mode = SamplingMode(settings["mode"])

        # Define required and optional parameters for each mode
        mode_params = {
            SamplingMode.RANDOM: {"required": ["sample_size"], "optional": []},
            SamplingMode.DIVERSE: {"required": ["sample_size", "diversity_mode"], "optional": []},
            SamplingMode.AI_SEARCH: {"required": [], "optional": ["prompt"]},
        }

        # Check for required parameters
        if mode in mode_params:
            for param in mode_params[mode]["required"]:
                if param not in settings:
                    raise ValueError(
                        f"Missing required parameter '{param}' for {mode.value} sampling mode"
                    )
                settings[param] = settings[param]

            # Add optional parameters if present
            for param in mode_params[mode]["optional"]:
                if param in settings:
                    settings[param] = settings[param]
        else:
            raise ValueError(f"Sampling mode {mode.value} is not implemented")

        self.state_manager.update("sampling_settings", settings)

    def add_sampling_batch(self, batch_data: Dict[int, List[int]]) -> None:
        """Add sampling batch to state"""
        sampled_images = self.state_manager.get("sampled_images", {})

        for dataset_id, image_ids in batch_data.items():
            str_dataset_id = str(dataset_id)
            if str_dataset_id not in sampled_images:
                sampled_images[str_dataset_id] = []

            existing = set(sampled_images[str_dataset_id])
            new = set(image_ids)

            if existing & new:
                logger.warning(
                    f"Duplicates found in sampling batch for dataset {dataset_id}: {existing & new}"
                )
                sampled_images[str_dataset_id].extend(list(new - existing))
            else:
                sampled_images[str_dataset_id].extend(image_ids)

        self.state_manager.set("sampled_images", sampled_images)

    def get_sampled_images(self) -> Dict[str, List[int]]:
        """Get all sampled images"""
        return self.state_manager.get("sampled_images", {})
