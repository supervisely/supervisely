from collections import namedtuple

CheckpointInfo = namedtuple(
    "CheckpointInfo",
    [
        "app_name",
        "session_id",
        "session_path",
        "session_link",
        "task_type",
        "training_project_name",
        "artifacts",
    ],
)
