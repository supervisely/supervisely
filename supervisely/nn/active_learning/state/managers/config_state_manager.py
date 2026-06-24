from typing import List, Optional

from supervisely.annotation.obj_class import ObjClass
from supervisely.api.api import Api
from supervisely.nn.active_learning.state.managers.project_state_manager import (
    StateManager,
)


class ConfigStateManager:
    """Handles Active Learning configuration settings:
    - definitions (class and tags names)
    - user settings (annotators and reviewers)
    - gpu agents
    """

    def __init__(self, state_manager: StateManager, api: Api):
        self.state_manager = state_manager
        self.api = api

    @property
    def classes(self) -> List[ObjClass]:
        """Get classes names"""
        # return self.state_manager.get("classes", default=[])
        classes = self.state_manager.get("classes", default=[])
        return [ObjClass.from_json(obj_class) for obj_class in classes]

    @property
    def classes_names(self) -> List[str]:
        """Get classes names"""
        return [obj_class.name for obj_class in self.classes]

    def set_classes(self, classes: List[ObjClass]) -> None:
        """Set classes names"""
        if not all(isinstance(obj_class, ObjClass) for obj_class in classes):
            raise ValueError("All elements in classes must be of type ObjClass.")
        existing_classes = self.classes
        existing_classes_names = [o.name for o in existing_classes]
        new_classes = []
        for obj_class in classes:
            if obj_class.name not in existing_classes_names:
                new_classes.append(obj_class)
        all_classes = existing_classes + new_classes
        prepared_classes = [o.to_json() for o in all_classes]
        self.state_manager.set("classes", prepared_classes)

    # @property
    # def tags(self) -> List[str]:
    #     """Get tags names"""
    #     return self.state_manager.get("tags", default=[])

    # def set_tags(self, tags: List[str]) -> None:
    #     """Set tags names"""
    #     self.state_manager.set("tags", tags)

    @property
    def annotators(self) -> List[int]:
        """Get annotators IDs"""
        return self.state_manager.get("annotators", default=[])

    def set_annotators(self, annotators: List[int]) -> None:
        """Set annotators IDs"""
        self.state_manager.set("annotators", annotators)

    @property
    def reviewers(self) -> List[int]:
        """Get reviewers IDs"""
        return self.state_manager.get("reviewers", default=[])

    def set_reviewers(self, reviewers: List[int]) -> None:
        """Set reviewers IDs"""
        self.state_manager.set("reviewers", reviewers)

    @property
    def gpu_agent(self) -> Optional[int]:
        """Get GPU agent ID"""
        return self.state_manager.get("gpu_agent", default=None)

    def set_gpu_agent(self, gpu_agent: int) -> None:
        """Set GPU agent ID"""
        self.state_manager.set("gpu_agent", gpu_agent)
