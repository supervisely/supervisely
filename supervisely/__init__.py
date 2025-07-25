# coding: utf-8
# isort: skip_file
import pkg_resources  # isort: skip
import os

try:
    __version__ = pkg_resources.require("supervisely")[0].version
except TypeError as e:
    __version__ = "development"

from supervisely.sly_logger import (
    logger,
    ServiceType,
    EventType,
    add_logger_handler,
    add_default_logging_into_file,
    get_task_logger,
    change_formatters_default_values,
    LOGGING_LEVELS,
)

from supervisely.function_wrapper import (
    main_wrapper,
    function_wrapper,
    catch_silently,
    function_wrapper_nofail,
    function_wrapper_external_logger,
)

from supervisely.io import fs
from supervisely.io import env
from supervisely.io import json
from supervisely.io import network_exceptions
from supervisely.io.fs_cache import FileCache

from supervisely.imaging import image

# legacy
# import supervisely.imaging.video as imagevideo
from supervisely.imaging import color

from supervisely.task.paths import TaskPaths

from supervisely.task.progress import (
    epoch_float,
    Progress,
    report_import_finished,
    report_dtl_finished,
    report_dtl_verification_finished,
    report_metrics_training,
    report_metrics_validation,
    report_inference_finished,
)


import supervisely.project as project
import supervisely.api.constants as api_constants
from supervisely.project import read_project, get_project_class
from supervisely.project.download import download, download_async, download_fast
from supervisely.project.upload import upload
from supervisely.project.project import (
    Project,
    OpenMode,
    download_project,
    read_single_project,
    upload_project,
    Dataset,
)
from supervisely.project.project_meta import ProjectMeta

from supervisely.annotation.annotation import ANN_EXT, Annotation
from supervisely.annotation.label import Label
from supervisely.annotation.obj_class import ObjClass, ObjClassJsonFields
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_meta import TagMeta, TagValueType, TagApplicableTo
from supervisely.annotation.tag import Tag
from supervisely.annotation.tag_collection import TagCollection
from supervisely.annotation.tag_meta_collection import TagMetaCollection

from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.cuboid import Cuboid
from supervisely.geometry.point import Point
from supervisely.geometry.point_location import PointLocation
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.geometry.mask_3d import Mask3D
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.graph import GraphNodes, Node
from supervisely.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.cuboid_2d import Cuboid2d

from supervisely.geometry.helpers import geometry_to_bitmap
from supervisely.geometry.helpers import deserialize_geometry

from supervisely.export.pascal_voc import save_project_as_pascal_voc_detection

from supervisely.metric.metric_base import MetricsBase
from supervisely.metric.projects_applier import MetricProjectsApplier


from supervisely.metric.iou_metric import IoUMetric
from supervisely.metric.confusion_matrix_metric import ConfusionMatrixMetric
from supervisely.metric.precision_recall_metric import PrecisionRecallMetric
from supervisely.metric.classification_metrics import ClassificationMetrics
from supervisely.metric.map_metric import MAPMetric

from supervisely.worker_api.agent_api import AgentAPI
from supervisely.worker_api.chunking import (
    ChunkSplitter,
    ChunkedFileWriter,
    ChunkedFileReader,
)
import supervisely.worker_proto.worker_api_pb2 as api_proto

from supervisely.api.api import Api, UserSession, ApiContext
from supervisely.api import api
from supervisely.api.task_api import WaitingTimeExceeded, TaskFinishedWithError
from supervisely.project.project_type import ProjectType
from supervisely.project.project_settings import ProjectSettings
from supervisely.api.report_api import NotificationType
from supervisely.api.image_api import ImageInfo
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.api.workspace_api import WorkspaceInfo
from supervisely.api.team_api import TeamInfo
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.app_api import WorkflowSettings, WorkflowMeta
from supervisely.api.entities_collection_api import EntitiesCollectionInfo

from supervisely.cli import _handle_creds_error_to_console

from supervisely._utils import (
    rand_str,
    batched,
    batched_iter,
    get_bytes_hash,
    generate_names,
    ENTERPRISE,
    COMMUNITY,
    _dprint,
    take_with_default,
    get_string_hash,
    is_development,
    is_production,
    is_debug_with_sly_net,
    compress_image_url,
    get_datetime,
    get_readable_datetime,
    generate_free_name,
    setup_certificates,
    is_community,
    run_coroutine,
)

import supervisely._utils as utils
from supervisely.tiny_timer import TinyTimer

from supervisely.aug import aug
from supervisely.video_annotation.key_id_map import KeyIdMap

from supervisely.video_annotation.video_annotation import VideoAnnotation
from supervisely.video_annotation.video_object import VideoObject
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.video_annotation.video_figure import VideoFigure
from supervisely.video_annotation.frame import Frame
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.video_tag import VideoTag
from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.project.video_project import (
    VideoDataset,
    VideoProject,
    download_video_project,
    upload_video_project,
)
from supervisely.video import video

import supervisely.labeling_jobs.utils as lj

from supervisely.pointcloud import pointcloud
from supervisely.pointcloud_episodes import pointcloud_episodes
from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely.pointcloud_annotation.pointcloud_episode_annotation import (
    PointcloudEpisodeAnnotation,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame import (
    PointcloudEpisodeFrame,
)
from supervisely.pointcloud_annotation.pointcloud_episode_frame_collection import (
    PointcloudEpisodeFrameCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object import (
    PointcloudEpisodeObject,
)
from supervisely.pointcloud_annotation.pointcloud_episode_object_collection import (
    PointcloudEpisodeObjectCollection,
)
from supervisely.pointcloud_annotation.pointcloud_episode_tag import (
    PointcloudEpisodeTag,
)
from supervisely.pointcloud_annotation.pointcloud_episode_tag_collection import (
    PointcloudEpisodeTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_tag import PointcloudTag
from supervisely.pointcloud_annotation.pointcloud_tag_collection import (
    PointcloudTagCollection,
)
from supervisely.project.pointcloud_project import (
    PointcloudDataset,
    PointcloudProject,
    download_pointcloud_project,
    upload_pointcloud_project,
)
from supervisely.project.pointcloud_episode_project import (
    PointcloudEpisodeDataset,
    PointcloudEpisodeProject,
    download_pointcloud_episode_project,
    upload_pointcloud_episode_project,
)

from supervisely.pyscripts_utils import utils as ps
from supervisely.io import docker_utils

import supervisely.app as app
from supervisely.app.fastapi import Application
from supervisely.app.v1.app_service import AppService
import supervisely.nn as nn

from supervisely.decorators.profile import timeit
from supervisely.decorators.profile import update_fields
from supervisely.decorators.inference import (
    process_image_roi,
    process_image_sliding_window,
)

import supervisely.script as script
from supervisely.user.user import UserRoleName
from supervisely.io import github_utils as git

from supervisely.aug import imgaug_utils

import supervisely.volume as volume
from supervisely.volume_annotation.volume_annotation import VolumeAnnotation
from supervisely.volume_annotation.volume_object import VolumeObject
from supervisely.volume_annotation.volume_object_collection import VolumeObjectCollection
from supervisely.volume_annotation.volume_tag import VolumeTag
from supervisely.volume_annotation.volume_tag_collection import VolumeTagCollection
from supervisely.volume_annotation.volume_figure import VolumeFigure
from supervisely.volume_annotation.slice import Slice
from supervisely.volume_annotation.plane import Plane
from supervisely.project.volume_project import (
    VolumeDataset,
    VolumeProject,
    download_volume_project,
    upload_volume_project,
)

from supervisely.convert.converter import ImportManager
from supervisely.convert.base_converter import AvailableImageConverters, BaseConverter

from supervisely.geometry.bitmap import SkeletonizeMethod

import supervisely.team_files as team_files
import supervisely.output as output

# start monkey patching
import importlib
import inspect
from supervisely.task.progress import tqdm_sly
import tqdm

from supervisely import convert

_original_tqdm = tqdm.tqdm


def get_module_names_from_stack(is_reversed=False) -> list:
    frame_records = inspect.stack()
    module_names = []
    for frame_record in frame_records:
        module_name = frame_record.frame.f_globals["__name__"]
        module_names.append(module_name)
    if is_reversed:
        module_names.reverse()
    return module_names


module_names = get_module_names_from_stack(is_reversed=True)

for mname in module_names:  # list starts with "__main__"
    m = importlib.import_module(mname)
    if not hasattr(m, "tqdm"):
        continue
    else:
        if hasattr(m.tqdm, "tqdm"):
            if type(m.tqdm.tqdm) != tqdm_sly:
                m.tqdm.tqdm = tqdm_sly
        else:
            if type(m.tqdm) != tqdm_sly:
                m.tqdm = tqdm_sly

# end monkeypatching

from supervisely.io.exception_handlers import handle_exceptions
from supervisely.app.fastapi.subapp import Event

try:
    setup_certificates()
except Exception as e:
    logger.warn(f"Failed to setup certificates. Reason: {repr(e)}", exc_info=True)

# If new changes in Supervisely Python SDK require upgrade of the Supervisely instance
# set a new value for the environment variable MINIMUM_INSTANCE_VERSION_FOR_SDK, otherwise
# users can face compatibility issues, if the instance version is lower than the SDK version.
os.environ["MINIMUM_INSTANCE_VERSION_FOR_SDK"] = "6.14.00"
