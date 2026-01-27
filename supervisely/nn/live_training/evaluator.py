from .metrics import detection_metrics, segmentation_metrics

class Evaluator:
    def __init__(self, task_type):
        self.task_type = task_type
        
    def evalutate(self, image_id):
        metric = image_id 
        return metric