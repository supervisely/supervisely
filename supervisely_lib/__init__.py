# coding: utf-8

from supervisely_lib.sly_logger import logger, ServiceType, EventType, add_logger_handler, \
    add_default_logging_into_file, get_task_logger, change_formatters_default_values, LOGGING_LEVELS

from supervisely_lib.function_wrapper import main_wrapper, function_wrapper, catch_silently, function_wrapper_nofail, \
    function_wrapper_external_logger

from supervisely_lib.io import fs
from supervisely_lib.io import env
from supervisely_lib.io import json
from supervisely_lib.io import network_exceptions
from supervisely_lib.io.fs_cache import FileCache

from supervisely_lib.imaging import image
# legacy
# import supervisely_lib.imaging.video as imagevideo
from supervisely_lib.imaging import color

from supervisely_lib.task.paths import TaskPaths

from supervisely_lib.task.progress import epoch_float, Progress, report_import_finished, report_dtl_finished, \
    report_dtl_verification_finished, \
    report_metrics_training, report_metrics_validation, report_inference_finished

from supervisely_lib.project.project import Project, OpenMode, download_project, read_single_project, upload_project, \
    Dataset
from supervisely_lib.project.project_meta import ProjectMeta

from supervisely_lib.annotation.annotation import ANN_EXT, Annotation
from supervisely_lib.annotation.label import Label
from supervisely_lib.annotation.obj_class import ObjClass, ObjClassJsonFields
from supervisely_lib.annotation.obj_class_collection import ObjClassCollection
from supervisely_lib.annotation.tag_meta import TagMeta, TagValueType, TagApplicableTo
from supervisely_lib.annotation.tag import Tag
from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.annotation.tag_meta_collection import TagMetaCollection

from supervisely_lib.geometry.bitmap import Bitmap
from supervisely_lib.geometry.cuboid import Cuboid
from supervisely_lib.geometry.point import Point
from supervisely_lib.geometry.point_location import PointLocation
from supervisely_lib.geometry.polygon import Polygon
from supervisely_lib.geometry.polyline import Polyline
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.any_geometry import AnyGeometry
from supervisely_lib.geometry.multichannel_bitmap import MultichannelBitmap

from supervisely_lib.geometry.helpers import geometry_to_bitmap
from supervisely_lib.geometry.helpers import deserialize_geometry

from supervisely_lib.export.pascal_voc import save_project_as_pascal_voc_detection

from supervisely_lib.metric.metric_base import MetricsBase
from supervisely_lib.metric.projects_applier import MetricProjectsApplier

from supervisely_lib.metric.iou_metric import IoUMetric
from supervisely_lib.metric.confusion_matrix_metric import ConfusionMatrixMetric
from supervisely_lib.metric.precision_recall_metric import PrecisionRecallMetric
from supervisely_lib.metric.classification_metrics import ClassificationMetrics
from supervisely_lib.metric.map_metric import MAPMetric

from supervisely_lib.worker_api.agent_api import AgentAPI
from supervisely_lib.worker_api.chunking import ChunkSplitter, ChunkedFileWriter, ChunkedFileReader
import supervisely_lib.worker_proto.worker_api_pb2 as api_proto

from supervisely_lib.api.api import Api
from supervisely_lib.api import api
from supervisely_lib.api.task_api import WaitingTimeExceeded
from supervisely_lib.project.project_type import ProjectType
from supervisely_lib.api.report_api import NotificationType

from supervisely_lib._utils import rand_str, batched, get_bytes_hash, generate_names, ENTERPRISE, COMMUNITY, _dprint, \
    take_with_default, get_string_hash
from supervisely_lib.tiny_timer import TinyTimer

from supervisely_lib.aug import aug

from supervisely_lib.video_annotation.video_annotation import VideoAnnotation
from supervisely_lib.video_annotation.video_object import VideoObject
from supervisely_lib.video_annotation.video_object_collection import VideoObjectCollection
from supervisely_lib.video_annotation.video_figure import VideoFigure
from supervisely_lib.video_annotation.frame import Frame
from supervisely_lib.video_annotation.frame_collection import FrameCollection
from supervisely_lib.project.video_project import VideoDataset, VideoProject, download_video_project, upload_video_project
from supervisely_lib.video import video

import supervisely_lib.labeling_jobs.utils as lj

from supervisely_lib.pointcloud import pointcloud
from supervisely_lib.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely_lib.pointcloud_annotation.pointcloud_object import PointcloudObject
from supervisely_lib.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely_lib.project.pointcloud_project import PointcloudDataset, PointcloudProject, download_pointcloud_project

from supervisely_lib.pyscripts_utils import utils as ps
from supervisely_lib.io import docker_utils
import supervisely_lib.app as app
from supervisely_lib.app.app_service import AppService
import supervisely_lib.app.widgets

from supervisely_lib.decorators.profile import timeit
from supervisely_lib.decorators.profile import update_fields

import supervisely_lib.script as script
from supervisely_lib.user.user import UserRoleName
from supervisely_lib.io import github_utils as git

from supervisely_lib.aug import imgaug_utils
