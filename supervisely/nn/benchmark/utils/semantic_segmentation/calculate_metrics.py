from supervisely.nn.benchmark.utils.semantic_segmentation.evaluator import Evaluator
from supervisely.nn.benchmark.utils.semantic_segmentation.loader import (
    build_segmentation_loader,
)


def calculate_metrics(
    gt_dir,
    pred_dir,
    boundary_width,
    boundary_iou_d,
    num_workers,
    class_names,
    result_dir,
    progress=None,
):
    if boundary_width % 1 == 0:
        boundary_width = int(boundary_width)
    evaluator = Evaluator(
        class_names=class_names,
        boundary_width=boundary_width,
        boundary_implementation="exact",
        boundary_iou_d=boundary_iou_d,
        result_dir=result_dir,
        progress=progress,
    )
    loader = build_segmentation_loader(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        gt_label_map=None,
        pred_label_map=None,
        num_workers=num_workers,
    )
    eval_data = evaluator.evaluate(loader)
    return eval_data
