# coding: utf-8

from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.annotation.obj_class import ObjClass
from supervisely_lib.annotation.tag import Tag
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.point import Point
from supervisely_lib.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib._utils import take_with_default


class LabelJsonFields:
    OBJ_CLASS_NAME = 'classTitle'
    DESCRIPTION = 'description'
    TAGS = 'tags'


class LabelBase:
    def __init__(self, geometry: Geometry, obj_class: ObjClass, tags: TagCollection = None, description: str = ""):
        self._geometry = geometry
        self._obj_class = obj_class
        self._tags = TagCollection() if tags is None else tags
        self._description = description
        self._validate_geometry_type()

    def _validate_geometry_type(self):
        raise NotImplementedError()

    @property
    def obj_class(self):
        return self._obj_class

    @property
    def geometry(self):
        return self._geometry
      
    @property
    def description(self):
        return self._description

    @property
    def tags(self):
        return self._tags.clone()

    def to_json(self):
        return {
            LabelJsonFields.OBJ_CLASS_NAME: self.obj_class.name,
            LabelJsonFields.DESCRIPTION: self.description,
            LabelJsonFields.TAGS: self.tags.to_json(),
            ** self.geometry.to_json()
        }

    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta):
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        return cls(geometry=obj_class.geometry_type.from_json(data),
                   obj_class=obj_class,
                   tags=TagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.obj_tag_metas),
                   description=data.get(LabelJsonFields.DESCRIPTION, ""))

    def add_tag(self, tag: Tag):
        return self.clone(tags=self._tags.add(tag))

    def add_tags(self, tags: list):
        return self.clone(tags=self._tags.add_items(tags))

    def clone(self, geometry: Geometry = None, obj_class: ObjClass = None, tags: TagCollection = None,
              description: str = ""):
        return self.__class__(geometry=take_with_default(geometry, self.geometry),
                              obj_class=take_with_default(obj_class, self.obj_class),
                              tags=take_with_default(tags, self.tags),
                              description=description or self.description)

    def crop(self, rect):
        # TODO: remove this hack: separate point class.
        def contains(rect, geometry):
            if isinstance(geometry, Point):
                return rect.contains_point(geometry)
            else:
                return rect.contains(geometry.to_bbox())

        return [self] if contains(rect, self.geometry) else [
            self.clone(geometry=g) for g in self.geometry.crop(rect)]

    def relative_crop(self, rect):
        return [self.clone(geometry=g) for g in self.geometry.relative_crop(rect)]

    def rotate(self, rotator):
        return self.clone(geometry=self.geometry.rotate(rotator))

    def resize(self, in_size, out_size):
        return self.clone(geometry=self.geometry.resize(in_size, out_size))

    def scale(self, factor):
        return self.clone(geometry=self.geometry.scale(factor))

    def translate(self, drow, dcol):
        return self.clone(geometry=self.geometry.translate(drow=drow, dcol=dcol))

    def fliplr(self, img_size):
        return self.clone(geometry=self.geometry.fliplr(img_size))

    def flipud(self, img_size):
        return self.clone(geometry=self.geometry.flipud(img_size))

    def _draw_tags(self, bitmap, font):
        bbox = self.geometry.to_bbox()
        texts = [tag.get_compact_str() for tag in self.tags]
        sly_image.draw_text_sequence(bitmap=bitmap,
                                     texts=texts,
                                     anchor_point=(bbox.top, bbox.left),
                                     corner_snap=sly_image.CornerAnchorMode.BOTTOM_LEFT,
                                     font=font)

    def draw(self, bitmap, thickness=1, draw_tags=False, tags_font=None):
        self.geometry.draw(bitmap, self.obj_class.color, thickness)
        if draw_tags:
            self._draw_tags(bitmap, tags_font)

    def draw_contour(self, bitmap, thickness=1, draw_tags=False, tags_font=None):
        self.geometry.draw_contour(bitmap, self.obj_class.color, thickness)
        if draw_tags:
            self._draw_tags(bitmap, tags_font)

    @property
    def area(self):
        return self.geometry.area


class Label(LabelBase):
    def _validate_geometry_type(self):
        if type(self._geometry) is not self._obj_class.geometry_type:
            raise RuntimeError("Input geometry type {!r} != geometry type of ObjClass {}"
                               .format(type(self._geometry), self._obj_class.geometry_type))


class PixelwiseScoresLabel(LabelBase):
    def _validate_geometry_type(self):
        if type(self._geometry) is not MultichannelBitmap:
            raise RuntimeError("Input geometry type {!r} != geometry type of ObjClass {}"
                               .format(type(self._geometry), MultichannelBitmap))