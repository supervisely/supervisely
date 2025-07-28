from supervisely.nn.tracker.base_tracker import BaseTracker
from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT
from supervisely.nn.tracker.botsort.tracker.annotation import Annotaion


class BotSortTracker(BaseTracker):

    def __init__(self, reid_model="osnet_1_0.pt", device: str = None, settings: dict = None):
        super().__init__(settings)
        self.tracker = BoTSORT(args=settings)
        self.settings = settings
    
    def update(self, frame: np.ndarray, annotation: Annotaion) -> VideoAnnotation:
        for tracker in self.trackers:
            # 1. convert detections to the format expected by the tracker
            tracker.update(frame, annotation)
        return self.result_video_annotation

    def reset(self):
        self.tracker.reset()

    def track(self, frames: list, annotations: list) -> VideoAnnotation:
        for frame, annotation in zip(frames, annotations):
            self.update(frame, annotation)