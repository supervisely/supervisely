from supervisely_lib.video_annotation.video_figure import VideoFigure


class VolumeFigure(VideoFigure):
    def validate_bounds(self, img_size, _auto_correct=False):
        raise NotImplementedError("Volumes do not support it yet")
