"""
Live Training core module for online learning apps.

Provides framework-agnostic components for:
- Loss plateau detection
- Request queue management
- Checkpoint resolution and downloading
- Artifacts uploading and report generation

Works with both MMDetection and MMSegmentation.

Note: DatasetCheckpointHook must be implemented in each app
due to mmengine dependency and framework-specific registration.
"""

from .loss_plateau_detector import LossPlateauDetector
from .request_queue import RequestQueue, RequestType
from .checkpoint_utils import (
    resolve_checkpoint,
    download_checkpoint_from_team_files,
)
from .artifacts_uploader import OnlineTrainingArtifactsUploader

__all__ = [
    'LossPlateauDetector',
    'RequestQueue',
    'RequestType',
    'resolve_checkpoint',
    'download_checkpoint_from_team_files',
    'OnlineTrainingArtifactsUploader',
]