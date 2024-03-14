from typing import List, Literal, NamedTuple
from supervisely.api.api import Api
from supervisely.api.file_api import FileInfo


class CheckpointInfo(NamedTuple):
    app_name: str
    session_id: int
    session_path: str
    session_link: str
    task_type: str
    training_project_name: str
    checkpoints: List[FileInfo]


import supervisely.nn.checkpoints.detectron2 as detectron2
import supervisely.nn.checkpoints.hrda as hrda
import supervisely.nn.checkpoints.mmclassification as mmclassification
import supervisely.nn.checkpoints.mmdetection as mmdetection
import supervisely.nn.checkpoints.mmdetection3 as mmdetection3
import supervisely.nn.checkpoints.mmsegmentation as mmsegmentation
import supervisely.nn.checkpoints.ritm as ritm
import supervisely.nn.checkpoints.unet as unet
import supervisely.nn.checkpoints.yolov5 as yolov5
import supervisely.nn.checkpoints.yolov5_v2 as yolov5_v2
import supervisely.nn.checkpoints.yolov8 as yolov8


def get_list(
    api: Api,
    team_id: int,
    framework: Literal[
        "yolov5",
        "yolov5_v2",
        "yolov8",
        "detectron2",
        "mmdetection",
        "mmdetection3",
        "mmsegmentation",
        "mmclassification",
        "ritm",
        "unet",
        "hrda",
    ],
) -> List[CheckpointInfo]:
    if framework == "yolov5":
        checkpoints = yolov5.get_list(api, team_id)
    elif framework == "yolov5_v2":
        checkpoints = yolov5_v2.get_list(api, team_id)
    elif framework == "yolov8":
        checkpoints = yolov8.get_list(api, team_id)
    elif framework == "detectron2":
        checkpoints = detectron2.get_list(api, team_id)
    elif framework == "mmdetection":
        checkpoints = mmdetection.get_list(api, team_id)
    elif framework == "mmdetection3":
        checkpoints = mmdetection3.get_list(api, team_id)
    elif framework == "mmsegmentation":
        checkpoints = mmsegmentation.get_list(api, team_id)
    elif framework == "mmclassification":
        checkpoints = mmclassification.get_list(api, team_id)
    elif framework == "ritm":
        checkpoints = ritm.get_list(api, team_id)
    elif framework == "unet":
        checkpoints = unet.get_list(api, team_id)
    elif framework == "hrda":
        checkpoints = hrda.get_list(api, team_id)
    else:
        raise NotImplementedError(f"Unknown framework: {framework}")

    return checkpoints
