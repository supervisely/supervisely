class TaskType:
    OBJECT_DETECTION = "object detection"
    INSTANCE_SEGMENTATION = "instance segmentation"
    SEMANTIC_SEGMENTATION = "semantic segmentation"
    POSE_ESTIMATION = "pose estimation"
    SALIENT_OBJECT_SEGMENTATION = "salient object segmentation"
    PROMPT_BASED_OBJECT_DETECTION = "prompt-based object detection"
    INTERACTIVE_SEGMENTATION = "interactive segmentation"
    PROMPTABLE_SEGMENTATION = "promptable segmentation"
    TRACKING = "tracking"
    OBJECT_DETECTION_3D = "object detection 3d"

AVAILABLE_TASK_TYPES = [
    TaskType.OBJECT_DETECTION,
    TaskType.INSTANCE_SEGMENTATION,
    TaskType.SEMANTIC_SEGMENTATION,
    TaskType.POSE_ESTIMATION,
    TaskType.SALIENT_OBJECT_SEGMENTATION,
    TaskType.PROMPT_BASED_OBJECT_DETECTION,
    TaskType.INTERACTIVE_SEGMENTATION,
    TaskType.PROMPTABLE_SEGMENTATION,
    TaskType.TRACKING,
    TaskType.OBJECT_DETECTION_3D,
]