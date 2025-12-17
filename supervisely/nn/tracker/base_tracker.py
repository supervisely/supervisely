from typing import List, Dict, Any
import supervisely as sly
from supervisely import Annotation, VideoAnnotation
import numpy as np

class BaseTracker:

    def __init__(self, settings: dict = None, device: str = None):
        import torch  # pylint: disable=import-error
        self.settings = settings or {}
        auto_device = "cuda" if torch.cuda.is_available() else "cpu"
        settings_device = self.settings.get("device")
        
        if settings_device is not None:
            if settings_device == "auto":
                self.device = auto_device
            else:
                self.device = settings_device
        else:
            self.device = device if device is not None else auto_device
                
        self._validate_device()


    def update(self, frame: np.ndarray, annotation: Annotation) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self) -> None:
        """Reset tracker state."""
        pass    
    
    def track(self, frames: List[np.ndarray], annotations: List[Annotation]) -> VideoAnnotation:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @property
    def video_annotation(self) -> VideoAnnotation:
        """Return the accumulated VideoAnnotation."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Get default configurable parameters for this tracker.
        Must be implemented in subclass.
        """
        raise NotImplementedError(
            f"Method get_default_params() must be implemented in {cls.__name__}"
        )

    def _validate_device(self) -> None:
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError(
                f"Invalid device '{self.device}'. Supported devices are 'cpu' or 'cuda'."
            )   
