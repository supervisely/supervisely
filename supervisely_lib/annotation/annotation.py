# coding: utf-8
from __future__ import annotations

import json
import itertools
import numpy as np
from typing import List
import operator
import cv2
from copy import deepcopy
from PIL import Image

from supervisely_lib import logger
from supervisely_lib.annotation.label import Label
from supervisely_lib.annotation.obj_class_collection import ObjClassCollection
from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.imaging import image as sly_image
from supervisely_lib.imaging import font as sly_font
from supervisely_lib._utils import take_with_default
from supervisely_lib.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely_lib.geometry.bitmap import Bitmap
from supervisely_lib.geometry.polygon import Polygon
from supervisely_lib.io.fs import ensure_base_path

# for imgaug
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

ANN_EXT = '.json'


class AnnotationJsonFields:
    IMG_DESCRIPTION = 'description'
    IMG_SIZE = 'size'
    IMG_SIZE_WIDTH = 'width'
    IMG_SIZE_HEIGHT = 'height'
    IMG_TAGS = 'tags'
    LABELS = 'objects'
    CUSTOM_DATA = "customBigData"
    PROBABILITY_CLASSES = "probabilityClasses"
    PROBABILITY_LABELS = "probabilityLabels"
    IMAGE_ID = "imageId"


class Annotation:
    '''
    This is a class for creating and using annotations for images

    Attributes:
        img_size (tuple): size of the image
        labels (list): list of Label class objects
        img_tags (list): list of image tags
        img_description (str): image description
        pixelwise_scores_labels (list)
    '''
    def __init__(self, img_size, labels=None, img_tags=None, img_description="",
                 pixelwise_scores_labels=None, custom_data=None, image_id=None):
        '''
        The constructor for Annotation class.
        :param img_size(tuple): size of the image
        :param labels(list): list of Label class objects
        :param img_tags(list): list of image tags
        :param img_description(str): image description
        :param pixelwise_scores_labels(list)
        '''
        if not isinstance(img_size, (tuple, list)):
            raise TypeError('{!r} has to be a tuple or a list. Given type "{}".'.format('img_size', type(img_size)))
        self._img_size = tuple(img_size)
        self._img_description = img_description
        self._img_tags = take_with_default(img_tags, TagCollection())
        self._labels = []
        self._add_labels_impl(self._labels, take_with_default(labels, []))
        self._pixelwise_scores_labels = []  # @TODO: store pixelwise scores as usual geometry labels
        self._add_labels_impl(self._pixelwise_scores_labels, take_with_default(pixelwise_scores_labels, []))
        self._custom_data = take_with_default(custom_data, {})
        self._image_id = image_id

    @property
    def img_size(self):
        return deepcopy(self._img_size)

    @property
    def image_id(self):
        return self._image_id

    @property
    def labels(self) -> List[Label]:
        return self._labels.copy()

    @property
    def pixelwise_scores_labels(self):
        return self._pixelwise_scores_labels.copy()

    @property
    def img_description(self):
        return self._img_description

    @property
    def img_tags(self) -> TagCollection:
        return self._img_tags

    def to_json(self):
        '''
        The function to_json convert annotation to json format
        :return: annotation in json format
        '''
        res = {
            AnnotationJsonFields.IMG_DESCRIPTION: self.img_description,
            AnnotationJsonFields.IMG_SIZE: {
                AnnotationJsonFields.IMG_SIZE_HEIGHT: int(self.img_size[0]),
                AnnotationJsonFields.IMG_SIZE_WIDTH: int(self.img_size[1])
            },
            AnnotationJsonFields.IMG_TAGS: self.img_tags.to_json(),
            AnnotationJsonFields.LABELS: [label.to_json() for label in self.labels],
            AnnotationJsonFields.CUSTOM_DATA: self.custom_data
        }
        if len(self._pixelwise_scores_labels) > 0:
            # construct probability classes from labels
            prob_classes = {}
            for label in self._pixelwise_scores_labels:
                # @TODO: hotfix to save geometry as "multichannelBitmap" instead of "bitmap"; use normal classes
                prob_classes[label.obj_class.name] = label.obj_class.clone(geometry_type=MultichannelBitmap)

            # save probabilities
            probabilities = {
                AnnotationJsonFields.PROBABILITY_LABELS: [label.to_json() for label in self._pixelwise_scores_labels],
                AnnotationJsonFields.PROBABILITY_CLASSES: ObjClassCollection(list(prob_classes.values())).to_json()
            }
            res[AnnotationJsonFields.CUSTOM_DATA].update(probabilities)
        if self.image_id is not None:
            res[AnnotationJsonFields.IMAGE_ID] = self.image_id
        return res

    @classmethod
    def from_json(cls, data, project_meta):
        '''
        The function from_json convert annotation from json format to Annotation class object. If one of the labels
        of annotation in json format cannot be convert to Label class object it generate exception error.
        :param data: input annotation in json format
        :param project_meta: ProjectMeta class object
        :return: Annotation class object
        '''
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

        custom_data = data.get(AnnotationJsonFields.CUSTOM_DATA, {})
        prob_labels = None
        if AnnotationJsonFields.PROBABILITY_LABELS in custom_data and \
                AnnotationJsonFields.PROBABILITY_CLASSES in custom_data:

            prob_classes = ObjClassCollection.from_json(custom_data[AnnotationJsonFields.PROBABILITY_CLASSES])

            # @TODO: tony, maybe link with project meta (add probability classes???)
            prob_project_meta = ProjectMeta(obj_classes=prob_classes)
            prob_labels = [Label.from_json(label_json, prob_project_meta)
                           for label_json in custom_data[AnnotationJsonFields.PROBABILITY_LABELS]]

            custom_data.pop(AnnotationJsonFields.PROBABILITY_CLASSES)
            custom_data.pop(AnnotationJsonFields.PROBABILITY_LABELS)

        image_id = data.get(AnnotationJsonFields.IMAGE_ID, None)

        return cls(img_size=img_size,
                   labels=labels,
                   img_tags=TagCollection.from_json(data[AnnotationJsonFields.IMG_TAGS], project_meta.tag_metas),
                   img_description=data.get(AnnotationJsonFields.IMG_DESCRIPTION, ""),
                   pixelwise_scores_labels=prob_labels,
                   custom_data=custom_data,
                   image_id=image_id)

    @classmethod
    def load_json_file(cls, path, project_meta):
        '''
        The function load_json_file download json file and convert in to the Annotation class object
        :param path: the path to the input json file
        :param project_meta: ProjectMeta class object
        :return: Annotation class object
        '''
        with open(path) as fin:
            data = json.load(fin)
        return cls.from_json(data, project_meta)

    def clone(self, img_size=None, labels=None, img_tags=None, img_description=None,
              pixelwise_scores_labels=None, custom_data=None, image_id=None):
        '''
        The function clone make copy of the Annotation class object
        :return: Annotation class object
        '''
        return Annotation(img_size=take_with_default(img_size, self.img_size),
                          labels=take_with_default(labels, self.labels),
                          img_tags=take_with_default(img_tags, self.img_tags),
                          img_description=take_with_default(img_description, self.img_description),
                          pixelwise_scores_labels=take_with_default(pixelwise_scores_labels, self.pixelwise_scores_labels),
                          custom_data=take_with_default(custom_data, self.custom_data),
                          image_id = take_with_default(image_id, self.image_id)
                          )

    def _add_labels_impl(self, dest, labels):
        '''
        The function _add_labels_impl extend list of the labels of the current Annotation object
        :param dest: destination list of the Label class objects
        :param labels: list of the Label class objects to be added to the destination list
        :return: list of the Label class objects
        '''
        for label in labels:
            # TODO Reconsider silent automatic normalization, reimplement
            canvas_rect = Rectangle.from_size(self.img_size)
            dest.extend(label.crop(canvas_rect))

    def add_label(self, label):
        '''
        The function add_label add label to the current Annotation object and return the copy of the
        current  Annotation object
        :param label: Label class object to be added
        :return: Annotation class object with the new list of the Label class objects
        '''
        return self.add_labels([label])

    def add_labels(self, labels):
        '''
        The function add_labels extend list of the labels of the current Annotation object and return the copy of the
        current  Annotation object
        :param labels: list of the Label class objects to be added
        :return: Annotation class object with the new list of the Label class objects
        '''
        new_labels = []
        self._add_labels_impl(new_labels, labels)
        return self.clone(labels=[*self._labels, *new_labels])

    def delete_label(self, label):
        '''
        The function delete_label detele label from the current Annotation object and return the copy of the
        current  Annotation object. If there is no deleted label in current Annotation object it generate exception
        error(KeyError).
        :param label: Label class object to be delete
        :return: Annotation class object with the new list of the Label class objects
        '''
        retained_labels = [_label for _label in self._labels if _label != label]
        if len(retained_labels) == len(self._labels):
            raise KeyError('Trying to delete a non-existing label of class: {}'.format(label.obj_class.name))
        return self.clone(labels=retained_labels)

    def add_pixelwise_score_label(self, label):
        '''
        The function add_pixelwise_score_label add label to the pixelwise_scores_labels and return the copy of the
        current  Annotation object
        :param label: Label class object to be added
        :return: Annotation class object with the new list of the pixelwise_scores_labels
        '''
        return self.add_pixelwise_score_labels([label])

    def add_pixelwise_score_labels(self, labels):
        '''
        The function add_pixelwise_score_labels extend list of the labels of the pixelwise_scores_labels and return
        the copy of the current  Annotation object.
        :param labels: list of the Label class objects to be added
        :return: Annotation class object with the new list of the pixelwise_scores_labels
        '''
        new_labels = []
        self._add_labels_impl(new_labels, labels)
        return self.clone(pixelwise_scores_labels=[*self._pixelwise_scores_labels, *new_labels])

    def add_tag(self, tag):
        '''
        The function add_tag add tag to the current Annotation object and return the copy of the
        current  Annotation object
        :param tag: TagCollection class object to be added
        :return: Annotation class object with the new list of the tags
        '''
        return self.clone(img_tags=self._img_tags.add(tag))

    def add_tags(self, tags):
        '''
        The function add_tags add tags to the current Annotation object and return the copy of the
        current  Annotation object
        :param tags: list of the TagCollection class objects to be added
        :return: Annotation class object with the new list of the tags
        '''
        return self.clone(img_tags=self._img_tags.add_items(tags))

    def delete_tags_by_name(self, tag_names):
        '''
        The function delete_tags_by_name removes tags by their names from current Annotation object and return the copy
        of the current  Annotation object
        :param tag_names: list of the tag names to be delete
        :return: Annotation class object with the new list of the tags
        '''
        retained_tags = [tag for tag in self._img_tags.items() if tag.meta.name not in tag_names]
        return self.clone(img_tags=TagCollection(items=retained_tags))

    def delete_tag_by_name(self, tag_name):
        '''
        The function delete_tag_by_name removes tag by it name from current Annotation object and return the copy
        of the current  Annotation object
        :param tag_name: tag names to be delete
        :return: Annotation class object with the new list of the tags
        '''
        return self.delete_tags_by_name([tag_name])

    def delete_tags(self, tags):
        '''
        The function delete_tags removes tags from current Annotation object and return the copy of the current
        Annotation object
        :param tags: list of the TagCollection class objects to be deleted
        :return: Annotation class object with the new list of the tags
        '''
        return self.delete_tags_by_name({tag.meta.name for tag in tags})

    def delete_tag(self, tag):
        '''
        The function delete_tag remove tag from current Annotation object and return the copy of the current
        Annotation object
        :param tag: TagCollection class object to be deleted
        :return: Annotation class object with the new list of the tags
        '''
        return self.delete_tags_by_name([tag.meta.name])

    def transform_labels(self, label_transform_fn, new_size=None):
        '''
        The function transform_labels transform labels and change image size in current Annotation object and return the copy of the current
        Annotation object
        :param label_transform_fn: function for transform labels
        :param new_size: new image size
        :return: Annotation class object with new labels and image size
        '''
        def _do_transform_labels(src_labels, label_transform_fn):
            # long easy to debug
            # result = []
            # for label in src_labels:
            #     result.extend(label_transform_fn(label))
            # return result

            # short, hard-to-debug alternative
            return list(itertools.chain(*[label_transform_fn(label) for label in src_labels]))
        new_labels = _do_transform_labels(self._labels, label_transform_fn)
        new_pixelwise_scores_labels = _do_transform_labels(self._pixelwise_scores_labels, label_transform_fn)
        return self.clone(img_size=take_with_default(new_size, self.img_size), labels=new_labels,
                          pixelwise_scores_labels=new_pixelwise_scores_labels)

    def crop_labels(self, rect):
        '''
        The function crop_labels crops labels in current Annotation object and return the copy of the current
        Annotation object
        :param rect: Rectangle class object
        :return: Annotation class object with new labels
        '''
        def _crop_label(label):
            return label.crop(rect)
        return self.transform_labels(_crop_label)

    def relative_crop(self, rect):
        '''
        The function relative_crop crops labels and change image size in current Annotation object and return the copy of the current
        Annotation object
        :param rect: Rectangle class object
        :return: Annotation class object with new labels and image size
        '''
        def _crop_label(label):
            return label.relative_crop(rect)
        return self.transform_labels(_crop_label, rect.to_size())

    def rotate(self, rotator):
        '''
        The function rotates all labels of the current Annotation object and return the copy of the current
        Annotation object
        :param rotator: ImageRotator class object
        :return: Annotation class object with new(rotated) labels and image size
        '''
        def _rotate_label(label):
            return [label.rotate(rotator)]
        return self.transform_labels(_rotate_label, tuple(rotator.new_imsize))

    def resize(self, out_size):
        '''
        The function resize all labels of the current Annotation object and return the copy of the current
        Annotation object
        :param out_size: new image size
        :return: Annotation class object with new(resized) labels and image size
        '''
        def _resize_label(label):
            return [label.resize(self.img_size, out_size)]
        return self.transform_labels(_resize_label, out_size)

    def scale(self, factor):
        '''
        The function scale change scale of the current Annotation object with a given factor and return the copy of the
        current Annotation object
        :param factor: float scale parameter
        :return: Annotation class object with new scale
        '''
        def _scale_label(label):
            return [label.scale(factor)]
        result_size = (round(self.img_size[0] * factor), round(self.img_size[1] * factor))
        return self.transform_labels(_scale_label, result_size)

    def fliplr(self):
        '''
        The function flip the current Annotation object in horizontal and return the copy of the
        current Annotation object
        :return: Annotation class object
        '''
        def _fliplr_label(label):
            return [label.fliplr(self.img_size)]
        return self.transform_labels(_fliplr_label)

    def flipud(self):
        '''
        The function flip the current Annotation object in vertical and return the copy of the
        current Annotation object
        :return: Annotation class object
        '''
        def _flipud_label(label):
            return [label.flipud(self.img_size)]
        return self.transform_labels(_flipud_label)

    def _get_font(self):
        '''
        The function get size of font for image with given size
        :return: font for drawing
        '''
        return sly_font.get_font(font_size=sly_font.get_readable_font_size(self.img_size))

    def _draw_tags(self, bitmap):
        '''
        The function draws text labels on bitmap from left to right.
        :param bitmap: target image
        '''
        texts = [tag.get_compact_str() for tag in self.img_tags]
        sly_image.draw_text_sequence(bitmap, texts, (0, 0), sly_image.CornerAnchorMode.TOP_LEFT,
                                     font=self._get_font())

    def draw(self, bitmap, color=None, thickness=1, draw_tags=False):
        '''
        The function draws rectangle near each label on bitmap
        :param bitmap: target image (np.ndarray)
        :param color: [R, G, B]
        :param thickness: thickness of the drawing rectangle
        '''
        tags_font = None
        if draw_tags is True:
            tags_font = self._get_font()
        for label in self._labels:
            label.draw(bitmap, color=color, thickness=thickness, draw_tags=draw_tags, tags_font=tags_font)
        if draw_tags:
            self._draw_tags(bitmap)

    def draw_contour(self, bitmap, color=None, thickness=1, draw_tags=False):
        '''
        The function draws the figure contour on a given bitmap
        :param bitmap: target image (np.ndarray)
        :param color: [R, G, B]
        :param thickness: thickness of the drawing contour
        '''
        tags_font = None
        if draw_tags is True:
            tags_font = self._get_font()
        for label in self._labels:
            label.draw_contour(bitmap, color=color, thickness=thickness, draw_tags=draw_tags, tags_font=tags_font)
        if draw_tags:
            self._draw_tags(bitmap)

    @classmethod
    def from_img_path(cls, img_path):
        '''
        The function from_img_path download image on the given path and return size of the image
        :param img_path: the path to the input image
        :return: size of the image
        '''
        img = sly_image.read(img_path)
        img_size = img.shape[:2]
        return cls(img_size)

    @classmethod
    def stat_area(cls, render, names, colors):
        if len(names) != len(colors):
            raise RuntimeError("len(names) != len(colors) [{} != {}]".format(len(names), len(colors)))

        result = {}

        height, width = render.shape[:2]
        total_pixels = height * width

        channels = None
        if len(render.shape) == 2:
            channels = 1
        elif len(render.shape) == 3:
            channels = render.shape[2]

        unlabeled_done = False

        covered_pixels = 0
        for name, color in zip(names, colors):
            col_name = name
            if name == "unlabeled":
                unlabeled_done = True
            class_mask = np.all(render == color, axis=-1).astype('uint8')
            cnt_pixels = class_mask.sum()
            covered_pixels += cnt_pixels
            result[col_name] = cnt_pixels / total_pixels * 100.0

        if covered_pixels > total_pixels:
            raise RuntimeError("Class colors mistake: covered_pixels > total_pixels")

        if unlabeled_done is False:
            result['unlabeled'] = (total_pixels - covered_pixels) / total_pixels * 100.0

        result['height'] = height
        result['width'] = width
        result['channels'] = channels
        return result

    def stat_class_count(self, class_names):
        total = 0
        stat = {name: 0 for name in class_names}
        for label in self._labels:
            cur_name = label.obj_class.name
            if cur_name not in stat:
                raise KeyError("Class {!r} not found in {}".format(cur_name, class_names))
            stat[cur_name] += 1
            total += 1
        stat['total'] = total
        return stat

    def draw_class_idx_rgb(self, render, name_to_index):
        for label in self._labels:
            class_idx = name_to_index[label.obj_class.name]
            color = [class_idx, class_idx, class_idx]
            label.draw(render, color=color, thickness=1)

    @property
    def custom_data(self):
        return self._custom_data.copy()

    def filter_labels_by_min_side(self, thresh, filter_operator=operator.lt, classes=None):
        def filter(label):
            if classes == None or label.obj_class.name in classes:
                bbox = label.geometry.to_bbox()
                height_px = bbox.height
                width_px = bbox.width
                if filter_operator(min(height_px, width_px), thresh):
                    return []  # action 'delete'
            return [label]
        return self.transform_labels(filter)

    def get_label_by_id(self, sly_id) -> Label:
        for label in self._labels:
            if label.geometry.sly_id == sly_id:
                return label
        return None

    def merge(self, other: Annotation):
        res = self.clone()
        res = res.add_labels(other.labels)
        res = res.add_tags(other.img_tags)
        return res

    def draw_pretty(self, bitmap: np.ndarray, color: List[int, int, int] = None, thickness: int = 1,
                    opacity: float = 0.5, draw_tags: bool = False, output_path: str = None) -> None:
        height, width = bitmap.shape[:2]
        vis_filled = np.zeros((height, width, 3), np.uint8)
        self.draw(vis_filled, color=color, thickness=thickness, draw_tags=draw_tags)
        vis = cv2.addWeighted(bitmap, 1, vis_filled, opacity, 0)
        np.copyto(bitmap, vis)
        if thickness > 0:
            self.draw_contour(bitmap, color=color, thickness=thickness, draw_tags=draw_tags)
        if output_path:
            sly_image.write(output_path, bitmap)

    def to_nonoverlapping_masks(self, mapping):
        common_img = np.zeros(self.img_size, np.int32)  # size is (h, w)
        for idx, lbl in enumerate(self.labels, start=1):
            #if mapping[lbl.obj_class] is not None:
            lbl.draw(common_img, color=idx)
        #(unique, counts) = np.unique(common_img, return_counts=True)
        new_labels = []
        for idx, lbl in enumerate(self.labels, start=1):
            dest_class = mapping[lbl.obj_class]
            if dest_class is None:
                continue  # skip labels
            mask = common_img == idx
            if np.any(mask):  # figure may be entirely covered by others
                g = lbl.geometry
                new_mask = Bitmap(data=mask)
                new_lbl = lbl.clone(geometry=new_mask, obj_class=dest_class)
                new_labels.append(new_lbl)
        new_ann = self.clone(labels=new_labels)
        return new_ann

    def to_indexed_color_mask(self, mask_path, palette=Image.ADAPTIVE, colors=256):
        mask = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        for label in self.labels:
            label.geometry.draw(mask, label.obj_class.color)

        im = Image.fromarray(mask)
        im = im.convert("P", palette=palette, colors=colors)

        ensure_base_path(mask_path)
        im.save(mask_path)

    def to_segmentation_task(self):
        class_mask = {}
        for label in self.labels:
            if label.obj_class not in class_mask:
                class_mask[label.obj_class] = np.zeros(self.img_size, np.uint8)
            label.draw(class_mask[label.obj_class], color=255)
        new_labels = []
        for obj_class, white_mask in class_mask.items():
            mask = white_mask == 255
            bitmap = Bitmap(data=mask)
            new_labels.append(Label(geometry=bitmap, obj_class=obj_class))
        return self.clone(labels=new_labels)

    def to_detection_task(self, mapping):
        aux_mapping = mapping.copy()

        to_render_mapping = {}
        to_render_labels = []
        other_labels = []
        _polygons_to_bitmaps_classes = {}
        for lbl in self.labels:
            if type(lbl.geometry) in [Bitmap, Polygon]:
                to_render_labels.append(lbl)
                if type(lbl.geometry) is Polygon:
                    new_class = _polygons_to_bitmaps_classes.get(lbl.obj_class.name, None)
                    if new_class is None:
                        new_class = lbl.obj_class.clone(geometry_type=Bitmap)
                        _polygons_to_bitmaps_classes[lbl.obj_class.name] = new_class
                        aux_mapping[new_class] = aux_mapping[lbl.obj_class]
                    to_render_mapping[lbl.obj_class] = new_class

                else:
                    to_render_mapping[lbl.obj_class] = lbl.obj_class
            else:
                other_labels.append(lbl)
        ann_raster = self.clone(labels=to_render_labels)
        ann_raster = ann_raster.to_nonoverlapping_masks(to_render_mapping)
        new_labels = []
        for lbl in [*ann_raster.labels, *other_labels]:
            dest_class = aux_mapping[lbl.obj_class]
            if dest_class is None:
                continue  # skip labels
            if dest_class == lbl.obj_class:
                new_labels.append(lbl)
            else:
                bbox = lbl.geometry.to_bbox()
                new_lbl = lbl.clone(geometry=bbox, obj_class=dest_class)
                new_labels.append(new_lbl)
        new_ann = self.clone(labels=new_labels)
        return new_ann

    def masks_to_imgaug(self, class_to_index) -> SegmentationMapsOnImage:
        h = self.img_size[0]
        w = self.img_size[1]
        mask = np.zeros((h, w, 1), dtype=np.int32)

        for label in self.labels:
            label: Label
            if type(label.geometry) == Bitmap:
                label.draw(mask, class_to_index[label.obj_class.name])

        segmaps = None
        if np.any(mask):
            segmaps = SegmentationMapsOnImage(mask, shape=self.img_size)
        return segmaps

    def bboxes_to_imgaug(self):
        boxes = []
        for label in self.labels:
            if type(label.geometry) == Rectangle:
                rect: Rectangle = label.geometry
                boxes.append(
                    BoundingBox(x1=rect.left, y1=rect.top,
                                x2=rect.right, y2=rect.bottom,
                                label=label.obj_class.name)
                )
        bbs = None
        if len(boxes) > 0:
            bbs = BoundingBoxesOnImage(boxes, shape=self.img_size)
        return bbs

    @classmethod
    def from_imgaug(cls, img, ia_boxes=None, ia_masks=None,
                    index_to_class=None,
                    meta: ProjectMeta = None):
        if ((ia_boxes is not None) or (ia_masks is not None)) and meta is None:
            raise ValueError("Project meta has to be provided")

        labels = []
        if ia_boxes is not None:
            for ia_box in ia_boxes:
                obj_class = meta.get_obj_class(ia_box.label)
                if obj_class is None:
                    raise KeyError("Class {!r} not found in project meta".format(ia_box.label))
                lbl = Label(Rectangle(top=ia_box.y1, left=ia_box.x1, bottom=ia_box.y2, right=ia_box.x2), obj_class)
                labels.append(lbl)

        if ia_masks is not None:
            if index_to_class is None:
                raise ValueError("mapping from index to class name is needed to transform masks to SLY format")
            class_mask = ia_masks.get_arr()
            # mask = white_mask == 255
            (unique, counts) = np.unique(class_mask, return_counts=True)
            for index, count in zip(unique, counts):
                if index == 0:
                    continue
                mask = class_mask == index
                bitmap = Bitmap(data=mask[:, :, 0])
                restore_class = meta.get_obj_class(index_to_class[index])
                labels.append(Label(geometry=bitmap, obj_class=restore_class))

        return cls(img_size=img.shape[:2], labels=labels)

    def is_empty(self):
        if len(self.labels) == 0 and len(self.img_tags) == 0:
            return True
        else:
            return False

    def filter_labels_by_classes(self, keep_classes):
        new_labels = []
        for lbl in self.labels:
            if lbl.obj_class.name in keep_classes:
                new_labels.append(lbl.clone())
        return self.clone(labels=new_labels)