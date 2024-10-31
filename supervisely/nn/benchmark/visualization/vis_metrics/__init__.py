from supervisely.nn.benchmark.visualization.vis_metrics.confidence_distribution import (
    ConfidenceDistribution,
)
from supervisely.nn.benchmark.visualization.vis_metrics.confidence_score import (
    ConfidenceScore,
)
from supervisely.nn.benchmark.visualization.vis_metrics.confusion_matrix import (
    ConfusionMatrix,
)
from supervisely.nn.benchmark.visualization.vis_metrics.explorer_grid import (
    ExplorerGrid,
)
from supervisely.nn.benchmark.visualization.vis_metrics.f1_score_at_different_iou import (
    F1ScoreAtDifferentIOU,
)
from supervisely.nn.benchmark.visualization.vis_metrics.frequently_confused import (
    FrequentlyConfused,
)
from supervisely.nn.benchmark.visualization.vis_metrics.iou_distribution import (
    IOUDistribution,
)
from supervisely.nn.benchmark.visualization.vis_metrics.model_predictions import (
    ModelPredictions,
)
from supervisely.nn.benchmark.visualization.vis_metrics.outcome_counts import (
    OutcomeCounts,
)
from supervisely.nn.benchmark.visualization.vis_metrics.outcome_counts_per_class import (
    PerClassOutcomeCounts,
)
from supervisely.nn.benchmark.visualization.vis_metrics.overview import Overview
from supervisely.nn.benchmark.visualization.vis_metrics.percision_avg_per_class import (
    PerClassAvgPrecision,
)
from supervisely.nn.benchmark.visualization.vis_metrics.pr_curve import PRCurve
from supervisely.nn.benchmark.visualization.vis_metrics.pr_curve_by_class import (
    PRCurveByClass,
)
from supervisely.nn.benchmark.visualization.vis_metrics.precision import Precision
from supervisely.nn.benchmark.visualization.vis_metrics.recall import Recall
from supervisely.nn.benchmark.visualization.vis_metrics.recall_vs_precision import (
    RecallVsPrecision,
)
from supervisely.nn.benchmark.visualization.vis_metrics.reliability_diagram import (
    ReliabilityDiagram,
)

ALL_METRICS = (
    Overview,
    ExplorerGrid,
    ModelPredictions,
    OutcomeCounts,
    Recall,
    Precision,
    RecallVsPrecision,
    PRCurve,
    PRCurveByClass,
    ConfusionMatrix,
    FrequentlyConfused,
    IOUDistribution,
    ReliabilityDiagram,
    ConfidenceScore,
    F1ScoreAtDifferentIOU,
    ConfidenceDistribution,
    PerClassAvgPrecision,
    PerClassOutcomeCounts,
)
