# coding: utf-8

from supervisely_lib.video_annotation.video_figure import VideoFigure


class PointcloudFigure(VideoFigure):
    """
    Class for creating and using PointcloudFigure objects for videos
    """
    def __init__(self, parent_object, geometry, frame_index=None, key=None,
                 class_id=None, labeler_login=None, updated_at=None, created_at=None):
        """
        :param parent_object:  VideoObject class object
        :param geometry: Geometry class object (cuboid_3d only)
        :param frame_index: int
        :param key: uuid class object
        """
        super().__init__(parent_object, geometry, frame_index, key, class_id=class_id, labeler_login=labeler_login, updated_at=updated_at, created_at=created_at)
        #@TODO: validate geometry - allowed: only cuboid_3d + point_cloud

    def validate_bounds(self, img_size, _auto_correct=False):
        raise NotImplementedError()
