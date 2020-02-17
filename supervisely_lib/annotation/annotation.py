# coding: utf-8

import json
import itertools

from copy import deepcopy

from supervisely_lib import logger
from supervisely_lib.annotation.label import Label
from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.imaging import font as sly_font
from supervisely_lib._utils import take_with_default

ANN_EXT = '.json'


class AnnotationJsonFields:
    IMG_DESCRIPTION = 'description'
    IMG_SIZE = 'size'
    IMG_SIZE_WIDTH = 'width'
    IMG_SIZE_HEIGHT = 'height'
    IMG_TAGS = 'tags'
    LABELS = 'objects'


class Annotation:
    def __init__(self, img_size, labels=None, img_tags=None, img_description="", pixelwise_scores_labels=None):
        if not isinstance(img_size, (tuple, list)):
            raise TypeError('{!r} has to be a tuple or a list. Given type "{}".'.format('img_size', type(img_size)))
        self._img_size = tuple(img_size)
        self._img_description = img_description
        self._img_tags = take_with_default(img_tags, TagCollection())
        self._labels = []
        self._add_labels_impl(self._labels, take_with_default(labels, []))
        self._pixelwise_scores_labels = []      # This field is not serialized. @TODO: create another class AnnotationExtended???
        self._add_labels_impl(self._pixelwise_scores_labels, take_with_default(pixelwise_scores_labels, []))

    @property
    def img_size(self):
        return deepcopy(self._img_size)

    @property
    def labels(self):
        return self._labels.copy()

    @property
    def pixelwise_scores_labels(self):
        return self._pixelwise_scores_labels.copy()

    @property
    def img_description(self):
        return self._img_description

    @property
    def img_tags(self):
        return self._img_tags

    def to_json(self):
        return {
            AnnotationJsonFields.IMG_DESCRIPTION: self.img_description,
            AnnotationJsonFields.IMG_SIZE: {
                AnnotationJsonFields.IMG_SIZE_HEIGHT: int(self.img_size[0]),
                AnnotationJsonFields.IMG_SIZE_WIDTH: int(self.img_size[1])
            },
            AnnotationJsonFields.IMG_TAGS: self.img_tags.to_json(),
            AnnotationJsonFields.LABELS: [label.to_json() for label in self.labels]
        }

    @classmethod
    def from_json(cls, data, project_meta):
        img_size_dict = data[AnnotationJsonFields.IMG_SIZE]
        img_height = img_size_dict[AnnotationJsonFields.IMG_SIZE_HEIGHT]
        img_width = img_size_dict[AnnotationJsonFields.IMG_SIZE_WIDTH]
        img_size = (img_height, img_width)
        try:
            labels = [Label.from_json(label_json, project_meta) for label_json in data[AnnotationJsonFields.LABELS]]
        except Exception:
            logger.fatal('Failed to deserialize annotation from JSON format. One of the Label objects could not be '
                         'deserialized')
            raise
        return cls(img_size=img_size,
                   labels=labels,
                   img_tags=TagCollection.from_json(data[AnnotationJsonFields.IMG_TAGS], project_meta.tag_metas),
                   img_description=data.get(AnnotationJsonFields.IMG_DESCRIPTION, ""))

    @classmethod
    def load_json_file(cls, path, project_meta):
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta)

    def clone(self, img_size=None, labels=None, img_tags=None, img_description=None, pixelwise_scores_labels=None):
        return Annotation(img_size=take_with_default(img_size, self.img_size),
                          labels=take_with_default(labels, self.labels),
                          img_tags=take_with_default(img_tags, self.img_tags),
                          img_description=take_with_default(img_description, self.img_description),
                          pixelwise_scores_labels=take_with_default(pixelwise_scores_labels,
                                                                    self.pixelwise_scores_labels))

    def _add_labels_impl(self, dest, labels):
        for label in labels:
            # TODO Reconsider silent automatic normalization, reimplement
            canvas_rect = Rectangle.from_size(self.img_size)
            dest.extend(label.crop(canvas_rect))

    def add_label(self, label):
        return self.add_labels([label])

    def add_labels(self, labels):
        new_labels = []
        self._add_labels_impl(new_labels, labels)
        return self.clone(labels=[*self._labels, *new_labels])

    def delete_label(self, label):
        retained_labels = [_label for _label in self._labels if _label != label]
        if len(retained_labels) == len(self._labels):
            raise KeyError('Trying to delete a non-existing label of class: {}'.format(label.obj_class.name))
        return self.clone(labels=retained_labels)

    def add_pixelwise_score_label(self, label):
        return self.add_pixelwise_score_labels([label])

    def add_pixelwise_score_labels(self, labels):
        new_labels = []
        self._add_labels_impl(new_labels, labels)
        return self.clone(pixelwise_scores_labels=[*self._pixelwise_scores_labels, *new_labels])

    def add_tag(self, tag):
        return self.clone(img_tags=self._img_tags.add(tag))

    def add_tags(self, tags):
        return self.clone(img_tags=self._img_tags.add_items(tags))

    def delete_tags_by_name(self, tag_names):
        retained_tags = [tag for tag in self._img_tags.items() if tag.meta.name not in tag_names]
        return self.clone(img_tags=TagCollection(items=retained_tags))

    def delete_tag_by_name(self, tag_name):
        return self.delete_tags_by_name([tag_name])

    def delete_tags(self, tags):
        return self.delete_tags_by_name({tag.meta.name for tag in tags})

    def delete_tag(self, tag):
        return self.delete_tags_by_name([tag.meta.name])

    def transform_labels(self, label_transform_fn, new_size=None):
        def _do_transform_labels(src_labels, label_transform_fn):
            return list(itertools.chain(*[label_transform_fn(label) for label in src_labels]))
        new_labels = _do_transform_labels(self._labels, label_transform_fn)
        new_pixelwise_scores_labels = _do_transform_labels(self._pixelwise_scores_labels, label_transform_fn)
        return self.clone(img_size=take_with_default(new_size, self.img_size), labels=new_labels,
                          pixelwise_scores_labels=new_pixelwise_scores_labels)

    def crop_labels(self, rect):
        def _crop_label(label):
            return label.crop(rect)
        return self.transform_labels(_crop_label)

    def relative_crop(self, rect):
        def _crop_label(label):
            return label.relative_crop(rect)
        return self.transform_labels(_crop_label, rect.to_size())

    def rotate(self, rotator):
        def _rotate_label(label):
            return [label.rotate(rotator)]
        return self.transform_labels(_rotate_label, tuple(rotator.new_imsize))

    def resize(self, out_size):
        def _resize_label(label):
            return [label.resize(self.img_size, out_size)]
        return self.transform_labels(_resize_label, out_size)

    def scale(self, factor):
        def _scale_label(label):
            return [label.scale(factor)]
        result_size = (round(self.img_size[0] * factor), round(self.img_size[1] * factor))
        return self.transform_labels(_scale_label, result_size)

    def fliplr(self):
        def _fliplr_label(label):
            return [label.fliplr(self.img_size)]
        return self.transform_labels(_fliplr_label)

    def flipud(self):
        def _flipud_label(label):
            return [label.flipud(self.img_size)]
        return self.transform_labels(_flipud_label)

    def _get_font(self):
        return sly_font.get_font(font_size=sly_font.get_readable_font_size(self.img_size))

    def _draw_tags(self, bitmap):
        texts = [tag.get_compact_str() for tag in self.img_tags]
        sly_image.draw_text_sequence(bitmap, texts, (0, 0), sly_image.CornerAnchorMode.TOP_LEFT,
                                     font=self._get_font())

    def draw(self, bitmap, color=None, thickness=1, draw_tags=False):
        for label in self._labels:
            label.draw(bitmap, color=color, thickness=thickness, draw_tags=draw_tags, tags_font=self._get_font())
        if draw_tags:
            self._draw_tags(bitmap)

    def draw_contour(self, bitmap, color=None, thickness=1, draw_tags=False):
        for label in self._labels:
            label.draw_contour(
                bitmap, color=color, thickness=thickness, draw_tags=draw_tags, tags_font=self._get_font())
        if draw_tags:
            self._draw_tags(bitmap)

    @classmethod
    def from_img_path(cls, img_path):
        img = sly_image.read(img_path)
        img_size = img.shape[:2]
        return cls(img_size)
