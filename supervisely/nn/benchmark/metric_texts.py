from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

if TYPE_CHECKING:
    from supervisely.nn.benchmark.benchmark import Benchmark

from collections import namedtuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from supervisely._utils import camel_to_snake, rand_str
from supervisely.collection.str_enum import StrEnum
from supervisely.nn.benchmark import metric_provider

checkpoint_name = "YOLOv8-L (COCO 2017 val)"


markdown_overview = f"""# {checkpoint_name}

## Overview

- **Model**: [YOLOv8-L]()
- **Year**: 2023
- **Authors**: ultralytics
- **Task type**: object detection
- **Training dataset (?)**: COCO 2017 train
- **Model classes (?)**: (80): a, b, c, … (collapse)
- **Model weights (?)**: [/path/to/yolov8l.pt]()
- **License (?)**: AGPL-3.0
- [GitHub](https://github.com/ultralytics/ultralytics)
"""

markdown_key_metrics = """## Key Metrics

Here, we comprehensively assess the model's performance by presenting a broad set of metrics, including mAP (mean Average Precision), Precision, Recall, IoU (Intersection over Union), Classification Accuracy, Calibration Score, and Inference Speed.

- **Mean Average Precision (mAP)**: A comprehensive metric of detection performance. mAP calculates the <abbr title="{definitions.average_precision}">average precision</abbr> across all classes at different levels of <abbr title="{definitions.iou_threshold}">IoU thresholds</abbr> and precision-recall trade-offs. In other words, it evaluates the performance of a model by considering its ability to detect and localize objects accurately across multiple IoU thresholds and object categories.
- **Precision**: Precision indicates how often the model's predictions are actually correct when it predicts an object. This calculates the ratio of correct detections to the total number of detections made by the model.
- **Recall**: Recall measures the model's ability to find all relevant objects in a dataset. This calculates the ratio of correct detections to the total number of instances in a dataset.
- **Intersection over Union (IoU)**: IoU measures how closely predicted bounding boxes match the actual (ground truth) bounding boxes. It is calculated as the area of overlap between the predicted bounding box and the ground truth bounding box, divided by the area of union of these bounding boxes.
- **Classification Accuracy**: We separately measure the model's capability to correctly classify objects. It’s calculated as a proportion of correctly classified objects among all matched detections. A predicted bounding box is considered matched if it overlaps a ground true bounding box with IoU equal or higher than 0.5.
- **Calibration Score**: This score represents the consistency of predicted probabilities (or <abbr title="{definitions.confidence_score}">confidence scores</abbr>) made by the model, evaluating how well the predicted probabilities align with actual outcomes. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
- **Inference Speed**: The number of frames per second (FPS) the model can process, measured with a batch size of 1. The inference speed is important in applications, where real-time object detection is required. Additionally, slower models pour more GPU resources, so their inference cost is higher.
"""

markdown_outcome_counts = """## Outcome Counts

This chart is used to evaluate the overall model performance by breaking down all predictions into <abbr title="{definitions.true_positives}">True Positives</abbr> (TP), <abbr title="{definitions.false_positives}">False Positives</abbr> (FP), and <abbr title="{definitions.false_negatives}">False Negatives</abbr> (FN). This helps to visually assess the type of errors the model often encounters.
"""
