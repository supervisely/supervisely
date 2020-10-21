# coding: utf-8

from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.annotation.obj_class import ObjClass
from supervisely_lib.geometry.any_geometry import AnyGeometry
from supervisely_lib.annotation.tag import Tag
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib._utils import take_with_default
from supervisely_lib.annotation.json_geometries_map import GET_GEOMETRY_FROM_STR
from supervisely_lib.geometry.constants import GEOMETRY_TYPE, GEOMETRY_SHAPE


class LabelJsonFields:
    OBJ_CLASS_NAME = 'classTitle'
    DESCRIPTION = 'description'
    TAGS = 'tags'


class LabelBase:
    '''
    This is a class for creating and using labeling objects for annotations

    Attributes:
        geometry: Geometry class of the object(point, rectangle, polygon, bitmap, line)
        obj_class: Class of objects (person, car, etc) with necessary properties: name, type of geometry (Polygon, Rectangle, ...)
                   and RGB color. Only one class can be associated with Label.
        tags: TagCollection object
        description(str): description of the label
    '''
    def __init__(self, geometry: Geometry, obj_class: ObjClass, tags: TagCollection = None, description: str = ""):
        '''
        :param geometry: Geometry class of the object(point, rectangle, polygon, bitmap, line)
        :param obj_class: Class of objects (person, car, etc) with necessary properties: name, type of geometry (Polygon, Rectangle, ...)
                   and RGB color. Only one class can be associated with Label.
        :param tags: TagCollection object
        :param description: description of the label
        '''
        self._geometry = geometry
        self._obj_class = obj_class
        self._tags = take_with_default(tags, TagCollection())
        self._description = description
        self._validate_geometry_type()
        self._validate_geometry()

    def _validate_geometry(self):
        '''
        The function checks the name of the Object for compliance.
        :return: generate ValueError error if name is mismatch
        '''
        self._geometry.validate(self._obj_class.geometry_type.geometry_name(), self.obj_class.geometry_config)

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
        '''
        The function to_json convert label to json format
        :return: Label in json format
        '''

        res = {
            LabelJsonFields.OBJ_CLASS_NAME: self.obj_class.name,
            LabelJsonFields.DESCRIPTION: self.description,
            LabelJsonFields.TAGS: self.tags.to_json(),
            ** self.geometry.to_json(),
            GEOMETRY_TYPE: self.geometry.geometry_name(),
            GEOMETRY_SHAPE: self.geometry.geometry_name(),
        }
        return res

    @classmethod
    def from_json(cls, data, project_meta: ProjectMeta):
        '''
        The function from_json convert Label from json format to Label class object. If there is no ObjClass from
        input json format in ProjectMeta, it generate RuntimeError error.
        :param data: input label in json format
        :param project_meta: ProjectMeta class object
        :return: Label class object
        '''
        obj_class_name = data[LabelJsonFields.OBJ_CLASS_NAME]
        obj_class = project_meta.get_obj_class(obj_class_name)
        if obj_class is None:
            raise RuntimeError(f'Failed to deserialize a Label object from JSON: label class name {obj_class_name!r} '
                               f'was not found in the given project meta.')

        if obj_class.geometry_type is AnyGeometry:
            geometry_type_actual = GET_GEOMETRY_FROM_STR(data[GEOMETRY_TYPE] if GEOMETRY_TYPE in data else data[GEOMETRY_SHAPE])
            geometry = geometry_type_actual.from_json(data)
        else:
            geometry = obj_class.geometry_type.from_json(data)

        return cls(geometry=geometry,
                   obj_class=obj_class,
                   tags=TagCollection.from_json(data[LabelJsonFields.TAGS], project_meta.tag_metas),
                   description=data.get(LabelJsonFields.DESCRIPTION, ""))

    def add_tag(self, tag: Tag):
        '''
        The function add_tag add tag to the current Label object and return the copy of the
        current Label object
        :param tag: TagCollection class object to be added
        :return: Label class object with the new list of the tags
        '''
        return self.clone(tags=self._tags.add(tag))

    def add_tags(self, tags: list):
        '''
        The function add_tag add tags to the current Label object and return the copy of the
        current Label object
        :param tags: list of the TagCollection class objects to be added
        :return: Label class object with the new list of the tags
        :param tags:
        :return:
        '''
        return self.clone(tags=self._tags.add_items(tags))

    def clone(self, geometry: Geometry = None, obj_class: ObjClass = None, tags: TagCollection = None,
              description: str = None):
        '''
        The function clone make copy of the Label class object
        :return: Label class object
        '''
        return self.__class__(geometry=take_with_default(geometry, self.geometry),
                              obj_class=take_with_default(obj_class, self.obj_class),
                              tags=take_with_default(tags, self.tags),
                              description=take_with_default(description, self.description))

    def crop(self, rect):
        '''
        The function crop the current geometry of Label
        :param rect: Rectangle class object
        :return: Label class object with new geometry
        '''
        if rect.contains(self.geometry.to_bbox()):
            return [self]
        else:
            # for compatibility of old slightly invalid annotations, some of them may be out of image bounds.
            # will correct it automatically
            result_geometries = self.geometry.crop(rect)
            if len(result_geometries) == 1:
                result_geometries[0]._copy_creation_info_inplace(self.geometry)
                return [self.clone(geometry=result_geometries[0])]
            else:
                return [self.clone(geometry=g) for g in self.geometry.crop(rect)]

    def relative_crop(self, rect):
        '''
        The function relative_crop crop the current geometry of Label
        :param rect: Rectangle class object
        :return: Label class object with new geometry
        '''
        return [self.clone(geometry=g) for g in self.geometry.relative_crop(rect)]

    def rotate(self, rotator):
        '''
        The function rotate Label geometry and return the copy of the current Label object
        :param rotator: ImageRotator class object
        :return: Label class object with new(rotated) geometry
        '''
        return self.clone(geometry=self.geometry.rotate(rotator))

    def resize(self, in_size, out_size):
        '''
        The function resize Label geometry and return the copy of the current Label object
        :return: Label class object with new geometry
        '''
        return self.clone(geometry=self.geometry.resize(in_size, out_size))

    def scale(self, factor):
        '''
        The function scale change scale of the current Label object with a given factor
        :param factor: float scale parameter
        :return: Label class object with new geometry
        '''
        return self.clone(geometry=self.geometry.scale(factor))

    def translate(self, drow, dcol):
        '''
        The function translate shifts the object by a certain number of pixels and return the copy of the current Label object
        :param drow: horizontal shift
        :param dcol: vertical shift
        :return: Label class object with new geometry
        '''
        return self.clone(geometry=self.geometry.translate(drow=drow, dcol=dcol))

    def fliplr(self, img_size):
        '''
        The function fliplr the current Label object geometry in horizontal and return the copy of the
        current Label object
        :param img_size: size of the image
        :return: Label class object with new geometry
        '''
        return self.clone(geometry=self.geometry.fliplr(img_size))

    def flipud(self, img_size):
        '''
        The function fliplr the current Label object geometry in vertical and return the copy of the
        current Label object
        :param img_size: size of the image
        :return: Label class object with new geometry
        '''
        return self.clone(geometry=self.geometry.flipud(img_size))

    def _draw_tags(self, bitmap, font):
        '''
        The function _draw_tags text of the tags on bitmap from left to right.
        :param bitmap: target image (np.ndarray)
        :param font: size of the font
        '''
        bbox = self.geometry.to_bbox()
        texts = [tag.get_compact_str() for tag in self.tags]
        sly_image.draw_text_sequence(bitmap=bitmap,
                                     texts=texts,
                                     anchor_point=(bbox.top, bbox.left),
                                     corner_snap=sly_image.CornerAnchorMode.BOTTOM_LEFT,
                                     font=font)

    def draw(self, bitmap, color=None, thickness=1, draw_tags=False, tags_font=None):
        '''
        The function draws rectangle near label geometry on bitmap
        :param bitmap: target image (np.ndarray)
        :param color: [R, G, B]
        :param thickness: thickness of the drawing rectangle
        '''
        effective_color = take_with_default(color, self.obj_class.color)
        self.geometry.draw(bitmap, effective_color, thickness, config=self.obj_class.geometry_config)
        if draw_tags:
            self._draw_tags(bitmap, tags_font)

    def draw_contour(self, bitmap, color=None, thickness=1, draw_tags=False, tags_font=None):
        '''
        The function draws the figure contour on a given label geometry bitmap
        :param bitmap: target image (np.ndarray)
        :param color: [R, G, B]
        :param thickness: thickness of the drawing contour
        '''
        effective_color = take_with_default(color, self.obj_class.color)
        self.geometry.draw_contour(bitmap, effective_color, thickness, config=self.obj_class.geometry_config)
        if draw_tags:
            self._draw_tags(bitmap, tags_font)

    @property
    def area(self):
        '''
        :return: area of current geometry in Label object
        '''
        return self.geometry.area

    def convert(self, new_obj_class: ObjClass):
        labels = []
        geometries = self.geometry.convert(new_obj_class.geometry_type)
        for g in geometries:
            labels.append(self.clone(geometry=g, obj_class=new_obj_class))
        return labels


class Label(LabelBase):
    def _validate_geometry_type(self):
        '''
        Checks geometry type for correctness
        '''
        if self._obj_class.geometry_type != AnyGeometry:
            if type(self._geometry) is not self._obj_class.geometry_type:
                raise RuntimeError("Input geometry type {!r} != geometry type of ObjClass {}"
                                   .format(type(self._geometry), self._obj_class.geometry_type))


class PixelwiseScoresLabel(LabelBase):
    def _validate_geometry_type(self):
        '''
        Checks geometry type for correctness
        '''
        if type(self._geometry) is not MultichannelBitmap:
            raise RuntimeError("Input geometry type {!r} != geometry type of ObjClass {}"
                               .format(type(self._geometry), MultichannelBitmap))