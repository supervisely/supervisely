import supervisely.nn.benchmark.instance_segmentation.text_templates as vis_texts
from supervisely.nn.benchmark.cv_tasks import CVTask
from supervisely.nn.benchmark.object_detection.visualizer import (
    ObjectDetectionVisualizer,
)


class InstanceSegmentationVisualizer(ObjectDetectionVisualizer):
    def __init__(self, api, eval_results, workdir="./visualizations"):
        super().__init__(api, eval_results, workdir)

        self.vis_texts = vis_texts
        self._widgets = False
        self.ann_opacity = 0.7

    @property
    def cv_task(self):
        return CVTask.INSTANCE_SEGMENTATION