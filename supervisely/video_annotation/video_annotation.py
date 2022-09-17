# coding: utf-8
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Iterator
from copy import deepcopy
import uuid
from uuid import UUID
from supervisely.project.project_meta import ProjectMeta


from supervisely._utils import take_with_default
from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.video_annotation.video_object_collection import VideoObjectCollection
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.constants import (
    FRAMES,
    IMG_SIZE,
    IMG_SIZE_HEIGHT,
    IMG_SIZE_WIDTH,
    DESCRIPTION,
    FRAMES_COUNT,
    TAGS,
    OBJECTS,
    VIDEO_ID,
    KEY,
    VIDEOS_MAP,
    VIDEO_NAME,
)
from supervisely.video_annotation.key_id_map import KeyIdMap


class VideoAnnotation:
    """
    VideoAnnotation for a single video. :class:`VideoAnnotation<VideoAnnotation>` object is immutable.

    :param img_size: Size of the image (height, width).
    :type img_size: Tuple[int, int] or List[int, int]
    :param frames_count: Number of frames in VideoAnnotation.
    :type frames_count: int
    :param objects: VideoObjectCollection object.
    :type objects: VideoObjectCollection, optional
    :param frames: FrameCollection object.
    :type frames: FrameCollection, optional
    :param tags: VideoTagCollection object.
    :type tags: VideoTagCollection, optional
    :param description: Video description.
    :type description: str, optional
    :param key: UUID object.
    :type key: UUID, optional
    :raises: :class:`TypeError`, if img_size is not tuple or list
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        # Simple VideoAnnotation example
        height, width = 500, 700
        frames_count = 10
        video_ann = sly.VideoAnnotation((height, width), frames_count)
        print(video_ann.to_json())
        # Output: {
        #     "size": {
        #         "height": 500,
        #         "width": 700
        #     },
        #     "description": "",
        #     "key": "abef780b01ad4063b4b961ab2ba2f410",
        #     "tags": [],
        #     "objects": [],
        #     "frames": [],
        #     "framesCount": 10
        # }

        # More complex VideoAnnotation example

        height, width = 500, 700
        frames_count = 1
        # VideoObjectCollection
        obj_class_car = sly.ObjClass('car', sly.Rectangle)
        video_obj_car = sly.VideoObject(obj_class_car)
        objects = sly.VideoObjectCollection([video_obj_car])
        # FrameCollection
        fr_index = 7
        geometry = sly.Rectangle(0, 0, 100, 100)
        video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
        frame = sly.Frame(fr_index, figures=[video_figure_car])
        frames = sly.FrameCollection([frame])
        # VideoTagCollection
        meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
        from supervisely.video_annotation.video_tag import VideoTag
        vid_tag = VideoTag(meta_car, value='acura')
        from supervisely.video_annotation.video_tag_collection import VideoTagCollection
        video_tags = VideoTagCollection([vid_tag])
        # Description
        descr = 'car example'

        video_ann = sly.VideoAnnotation((height, width), frames_count, objects, frames, video_tags, descr)
        print(video_ann.to_json())
        # Output: {
        #     "size": {
        #         "height": 500,
        #         "width": 700
        #     },
        #     "description": "car example",
        #     "key": "a85b282e5e174e7ebad6f878b6919244",
        #     "tags": [
        #         {
        #             "name": "car_tag",
        #             "value": "acura",
        #             "key": "540a8212b0344788953996cea220ea8b"
        #         }
        #     ],
        #     "objects": [
        #         {
        #             "key": "7c74b8a495044ea0ac127f32751c8f5c",
        #             "classTitle": "car",
        #             "tags": []
        #         }
        #     ],
        #     "frames": [
        #         {
        #             "index": 7,
        #             "figures": [
        #                 {
        #                     "key": "82dcbf2e3c5f42a99eeea2ad34173793",
        #                     "objectKey": "7c74b8a495044ea0ac127f32751c8f5c",
        #                     "geometryType": "rectangle",
        #                     "geometry": {
        #                         "points": {
        #                             "exterior": [
        #                                 [
        #                                     0,
        #                                     0
        #                                 ],
        #                                 [
        #                                     100,
        #                                     100
        #                                 ]
        #                             ],
        #                             "interior": []
        #                         }
        #                     }
        #                 }
        #             ]
        #         }
        #     ],
        #     "framesCount": 1
        # }
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        frames_count: int,
        objects: Optional[VideoObjectCollection] = None,
        frames: Optional[FrameCollection] = None,
        tags: Optional[VideoTagCollection] = None,
        description: Optional[str] = "",
        key: Optional[UUID] = None,
    ):
        if not isinstance(img_size, (tuple, list)):
            raise TypeError(
                '{!r} has to be a tuple or a list. Given type "{}".'.format(
                    "img_size", type(img_size)
                )
            )
        self._img_size = tuple(img_size)
        self._frames_count = frames_count

        self._description = description
        self._tags = take_with_default(tags, VideoTagCollection())
        self._objects = take_with_default(objects, VideoObjectCollection())
        self._frames = take_with_default(frames, FrameCollection())
        self._key = take_with_default(key, uuid.uuid4())

        self.validate_figures_bounds()

    @property
    def img_size(self):
        """
        Size of the image (height, width).

        :return: Image size
        :rtype: :class:`Tuple[int, int]`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 1
            video_ann = sly.VideoAnnotation((height, width), frames_count)
            print(video_ann.img_size)
            # Output: (500, 700)
        """
        return deepcopy(self._img_size)

    @property
    def frames_count(self):
        """
        Number of frames.

        :return: Frames count
        :rtype: :class:`int`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 15
            video_ann = sly.VideoAnnotation((height, width), frames_count)
            print(video_ann.frames_count)
            # Output: 15
        """
        return self._frames_count

    @property
    def objects(self):
        """
        VideoAnnotation objects.

        :return: VideoObjectCollection object
        :rtype: :class:`VideoObjectCollection`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 1
            # VideoObjectCollection
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            objects = sly.VideoObjectCollection([video_obj_car])
            video_ann = sly.VideoAnnotation((height, width), frames_count, objects)
            print(video_ann.objects.to_json())
            # Output: [
            #     {
            #         "key": "79fc07a4a6ca4b2796279bc033b9ec9a",
            #         "classTitle": "car",
            #         "tags": []
            #     }
            # ]
        """
        return self._objects

    @property
    def frames(self):
        """
        VideoAnnotation frames.

        :return: FrameCollection object
        :rtype: :class:`FrameCollection`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 1
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            objects = sly.VideoObjectCollection([video_obj_car])
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
            frame = sly.Frame(fr_index, figures=[video_figure_car])
            frames = sly.FrameCollection([frame])

            video_ann = sly.VideoAnnotation((height, width), frames_count, objects, frames)
            print(video_ann.frames.to_json())
            # Output: [
            #     {
            #         "index": 7,
            #         "figures": [
            #             {
            #                 "key": "2842f561b1924f6abd6ab6f696ed9b65",
            #                 "objectKey": "7f30fa9b78444ad69e02b37edbf9a902",
            #                 "geometryType": "rectangle",
            #                 "geometry": {
            #                     "points": {
            #                         "exterior": [
            #                             [
            #                                 0,
            #                                 0
            #                             ],
            #                             [
            #                                 100,
            #                                 100
            #                             ]
            #                         ],
            #                         "interior": []
            #                     }
            #                 }
            #             }
            #         ]
            #     }
            # ]
        """
        return self._frames

    @property
    def figures(self):
        """
        VideoAnnotation figures.

        :return: List of VideoFigures from all frames in VideoAnnotation
        :rtype: :class:`List[VideoFigure]<supervisely.video_annotation.video_figure.VideoFigure>`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 1
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            objects = sly.VideoObjectCollection([video_obj_car])
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
            frame = sly.Frame(fr_index, figures=[video_figure_car])
            frames = sly.FrameCollection([frame])

            video_ann = sly.VideoAnnotation((height, width), frames_count, objects, frames)
            print(len(video_ann.figures)) # 1
        """
        return self.frames.figures

    @property
    def tags(self) -> VideoTagCollection:
        """
        VideoAnnotation tags.

        :return: VideoTagCollection object
        :rtype: :class:`VideoTagCollection`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 1
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            objects = sly.VideoObjectCollection([video_obj_car])
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
            frame = sly.Frame(fr_index, figures=[video_figure_car])
            frames = sly.FrameCollection([frame])
            meta_car = sly.TagMeta('car_tag', sly.TagValueType.ANY_STRING)
            from supervisely.video_annotation.video_tag import VideoTag
            vid_tag = VideoTag(meta_car, value='acura')
            from supervisely.video_annotation.video_tag_collection import VideoTagCollection
            tags = VideoTagCollection([vid_tag])

            video_ann = sly.VideoAnnotation((height, width), frames_count, objects, frames, tags)
            print(video_ann.tags.to_json())
            # Output: [
            #     {
            #         "name": "car_tag",
            #         "value": "acura",
            #         "key": "c63e8259589a4fa5b4fb15a48c1f6a63"
            #     }
            # ]
        """
        return self._tags

    def key(self):
        return self._key

    @property
    def description(self):
        """
        Video description.

        :return: Video description
        :rtype: :class:`str`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 1
            descr = 'example'
            video_ann = sly.VideoAnnotation((height, width), frames_count, description=descr)
            print(video_ann.description) # example
        """
        return self._description

    def validate_figures_bounds(self) -> None:
        """
        Checks if image contains figures from all frames in collection.

        :raises: :class:`OutOfImageBoundsExtension<supervisely.video_annotation.video_figure.OutOfImageBoundsExtension>`, if figure is out of image bounds
        :return: None
        :rtype: :class:`NoneType`

        :Usage Example:

         .. code-block:: python

            import supervisely as sly

            height, width = 50, 700
            frames_count = 1
            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            objects = sly.VideoObjectCollection([video_obj_car])
            fr_index = 7
            geometry = sly.Rectangle(0, 0, 100, 100)
            video_figure_car = sly.VideoFigure(video_obj_car, geometry, fr_index)
            frame = sly.Frame(fr_index, figures=[video_figure_car])
            frames = sly.FrameCollection([frame])

            video_ann = sly.VideoAnnotation((height, width), frames_count, objects, frames)
            video_ann.validate_figures_bounds()
            # raise OutOfImageBoundsExtension("Figure is out of image bounds")
        """
        for frame in self.frames:
            frame.validate_figures_bounds(self.img_size)

    def to_json(self, key_id_map: Optional[KeyIdMap] = None) -> Dict:
        """
        Convert the VideoAnnotation to a json dict. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: Json format as a dict
        :rtype: :class:`dict`
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 10
            video_ann = sly.VideoAnnotation((height, width), frames_count)
            print(video_ann.to_json())
            # Output: {
            #     "size": {
            #         "height": 500,
            #         "width": 700
            #     },
            #     "description": "",
            #     "key": "abef780b01ad4063b4b961ab2ba2f410",
            #     "tags": [],
            #     "objects": [],
            #     "frames": [],
            #     "framesCount": 10
            # }
        """
        res_json = {
            IMG_SIZE: {
                IMG_SIZE_HEIGHT: int(self.img_size[0]),
                IMG_SIZE_WIDTH: int(self.img_size[1]),
            },
            DESCRIPTION: self.description,
            KEY: self.key().hex,
            TAGS: self.tags.to_json(key_id_map),
            OBJECTS: self.objects.to_json(key_id_map),
            FRAMES: self.frames.to_json(key_id_map),
            FRAMES_COUNT: self.frames_count,
        }

        if key_id_map is not None:
            video_id = key_id_map.get_video_id(self.key())
            if video_id is not None:
                res_json[VIDEO_ID] = video_id

        return res_json

    @classmethod
    def from_json(
        cls, data: Dict, project_meta: ProjectMeta, key_id_map: Optional[KeyIdMap] = None
    ) -> VideoAnnotation:
        """
        Convert a json dict to VideoAnnotation. Read more about `Supervisely format <https://docs.supervise.ly/data-organization/00_ann_format_navi>`_.

        :param data: Dict in json format.
        :type data: dict
        :param project_meta: Input :class:`ProjectMeta<supervisely.project.project_meta.ProjectMeta>`.
        :type project_meta: ProjectMeta
        :param key_id_map: KeyIdMap object.
        :type key_id_map: KeyIdMap, optional
        :return: VideoAnnotation object
        :rtype: :class:`VideoAnnotation`

        :Usage example:

         .. code-block:: python

            import supervisely as sly

            video_ann_json = {
                "size": {
                    "height": 500,
                    "width": 700
                },
                "tags": [],
                "objects": [],
                "frames": [],
                "framesCount": 1
            }
            meta = sly.ProjectMeta()
            video_ann = sly.VideoAnnotation.from_json(video_ann_json, meta)
        """
        # video_name = data[VIDEO_NAME]
        video_key = uuid.UUID(data[KEY]) if KEY in data else uuid.uuid4()

        if key_id_map is not None:
            key_id_map.add_video(video_key, data.get(VIDEO_ID, None))

        img_size_dict = data[IMG_SIZE]
        img_height = img_size_dict[IMG_SIZE_HEIGHT]
        img_width = img_size_dict[IMG_SIZE_WIDTH]
        img_size = (img_height, img_width)

        description = data.get(DESCRIPTION, "")
        frames_count = data[FRAMES_COUNT]

        tags = VideoTagCollection.from_json(data[TAGS], project_meta.tag_metas, key_id_map)
        objects = VideoObjectCollection.from_json(data[OBJECTS], project_meta, key_id_map)
        frames = FrameCollection.from_json(data[FRAMES], objects, frames_count, key_id_map)

        return cls(
            img_size=img_size,
            frames_count=frames_count,
            objects=objects,
            frames=frames,
            tags=tags,
            description=description,
            key=video_key,
        )

    def clone(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        frames_count: Optional[int] = None,
        objects: Optional[VideoObjectCollection] = None,
        frames: Optional[FrameCollection] = None,
        tags: Optional[VideoTagCollection] = None,
        description: Optional[str] = None,
    ) -> VideoAnnotation:
        """
        Makes a copy of VideoAnnotation with new fields, if fields are given, otherwise it will use fields of the original VideoAnnotation.

        :param img_size: Size of the image (height, width).
        :type img_size: Tuple[int, int], optional
        :param frames_count: Number of frames in VideoAnnotation.
        :type frames_count: int, optional
        :param objects: VideoObjectCollection object.
        :type objects: VideoObjectCollection, optional
        :param frames: FrameCollection object.
        :type frames: FrameCollection, optional
        :param tags: VideoTagCollection object.
        :type tags: VideoTagCollection, optional
        :param description: Video description.
        :type description: str, optional
        :raises: :class:`TypeError`, if img_size is not tuple or list
        :Usage example:

         .. code-block:: python

            import supervisely as sly

            height, width = 500, 700
            frames_count = 1
            video_ann = sly.VideoAnnotation((height, width), frames_count)

            obj_class_car = sly.ObjClass('car', sly.Rectangle)
            video_obj_car = sly.VideoObject(obj_class_car)
            new_objects = sly.VideoObjectCollection([video_obj_car])
            new_video_ann = video_ann.clone(objects=new_objects)
            print(new_video_ann.to_json())
            # Output: {
            #     "size": {
            #         "height": 500,
            #         "width": 700
            #     },
            #     "description": "",
            #     "key": "37f7d267864c4fd8b1a1a32f67e37f7d",
            #     "tags": [],
            #     "objects": [
            #         {
            #             "key": "27d4ba1aaee64930b2d0bfb7e8b53493",
            #             "classTitle": "car",
            #             "tags": []
            #         }
            #     ],
            #     "frames": [],
            #     "framesCount": 1
            # }
        """
        return VideoAnnotation(
            img_size=take_with_default(img_size, self.img_size),
            frames_count=take_with_default(frames_count, self.frames_count),
            objects=take_with_default(objects, self.objects),
            frames=take_with_default(frames, self.frames),
            tags=take_with_default(tags, self.tags),
            description=take_with_default(description, self.description),
        )

    def is_empty(self) -> bool:
        """
        Check whether video annotation contains objects or tags, or not.

        :returns: True if video annotation is empty, False otherwise.
        :rtype: :class:`bool`
        """
        if len(self.objects) == 0 and len(self.tags) == 0:
            return True
        else:
            return False
