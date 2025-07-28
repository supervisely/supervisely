class BaseTracker:

    def __init__(self):
        self.settings = {}
        self.result_video_annotation = None

    def update(self, frame, detections):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self):
        self.result_video_annotation = None
    
    def track(self, frames: list, annoations: list):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _annotations_to_video_annotation(self, annotations: list) -> VideoAnnotation:
        """Convert a list of annotations to a VideoAnnotation object."""
        if not annotations:
            return VideoAnnotation()

        video_annotation = VideoAnnotation()
        for annotation in annotations:
            video_annotation.add(annotation)
        
        return video_annotation