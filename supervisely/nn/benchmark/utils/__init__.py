# isort: skip_file
from supervisely.nn.benchmark.utils.detection.calculate_metrics import calculate_metrics
from supervisely.nn.benchmark.utils.detection.sly2coco import sly2coco
from supervisely.nn.benchmark.utils.detection.utlis import read_coco_datasets
from supervisely.nn.benchmark.utils.detection.utlis import try_set_conf_auto

from supervisely.nn.benchmark.utils.semantic_segmentation.calculate_metrics import (
    calculate_metrics as calculate_semsegm_metrics,
)
from supervisely.nn.benchmark.utils.semantic_segmentation.evaluator import Evaluator
from supervisely.nn.benchmark.utils.semantic_segmentation.loader import build_segmentation_loader
