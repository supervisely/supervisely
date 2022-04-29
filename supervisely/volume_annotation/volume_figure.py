# from supervisely.video_annotation.video_figure import VideoFigure
# from supervisely.video_annotation.video_object_collection import (
#     VideoObjectCollection,
# )
# from supervisely.video_annotation.key_id_map import KeyIdMap

# # from sdk_part.volume_annotation.closed_surface_mesh import ClosedSurfaceMesh


# class VolumeFigure(VideoFigure):
#     def validate_bounds(self, img_size, _auto_correct=False):
#         raise NotImplementedError("Volumes do not support it yet")

#     @classmethod
#     def from_json(
#         cls, data, objects: VideoObjectCollection, key_id_map: KeyIdMap = None
#     ):
#         return super().from_json(data, objects, 0, key_id_map)

#     def _validate_geometry_type(self):

#         if (
#             type(self._geometry) != ClosedSurfaceMesh
#             and self.parent_object.obj_class.geometry_type != AnyGeometry
#         ):
#             if type(self._geometry) is not self.parent_object.obj_class.geometry_type:
#                 raise RuntimeError(
#                     "Input geometry type {!r} != geometry type of ObjClass {}".format(
#                         type(self._geometry), self.parent_object.obj_class.geometry_type
#                     )
#                 )

#     def _validate_geometry(self):
#         pass
