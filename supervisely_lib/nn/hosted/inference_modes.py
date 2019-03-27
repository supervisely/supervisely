# coding: utf-8

import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pkg_resources

from supervisely_lib.metric.common import safe_ratio
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.annotation.label import Label
from supervisely_lib.annotation.obj_class import ObjClass
from supervisely_lib.annotation.obj_class_collection import make_renamed_classes
from supervisely_lib.annotation.obj_class_mapper import RenamingObjClassMapper, replace_labels_classes
from supervisely_lib.annotation.tag_meta_collection import make_renamed_tag_metas
from supervisely_lib.annotation.tag_meta import TagValueType
from supervisely_lib.annotation.tag_meta_mapper import RenamingTagMetaMapper, make_renamed_tags
from supervisely_lib.annotation.renamer import Renamer, is_name_included
from supervisely_lib.geometry.geometry import Geometry
from supervisely_lib.geometry.multichannel_bitmap import MultichannelBitmap
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.sliding_windows import SlidingWindows
from supervisely_lib.nn import raw_to_labels
from supervisely_lib.nn.config import update_recursively, update_strict, MultiTypeValidator
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase


INFERENCE_MODE_CONFIG = 'inference_mode_config'
CLASS_NAME = 'class_name'
FROM_CLASSES = 'from_classes'
MODEL_CLASSES = 'model_classes'
MODE = 'mode'
NAME = 'name'
SAVE = 'save'

MATCH_ALL = '__all__'

WINDOW = 'window'
HEIGHT = 'height'
WIDTH = 'width'
MIN_OVERLAP = 'min_overlap'
X = 'x'
Y = 'y'

NMS_AFTER = 'nms_after'
ENABLE = 'enable'
IOU_THRESHOLD = 'iou_threshold'

# TODO move somewhere near geometric transformations.
BOUNDS = 'bounds'
PADDING = 'padding'

BOTTOM = 'bottom'
LEFT = 'left'
RIGHT = 'right'
TOP = 'top'

PERCENT = '%'
PX = 'px'

CONFIDENCE = 'confidence'
CONFIDENCE_TAG_NAME = 'confidence_tag_name'


def _rectangle_from_cropping_or_padding_bounds(img_shape, crop_config, do_crop: bool):
    def get_crop_pixels(raw_side, dim_name):
        side_crop_config = crop_config.get(dim_name)
        if side_crop_config is None:
            crop_pixels = 0
        elif side_crop_config.endswith(PX):
            crop_pixels = int(side_crop_config[:-len(PX)])
        elif side_crop_config.endswith(PERCENT):
            padding_fraction = float(side_crop_config[:-len(PERCENT)])
            crop_pixels = int(raw_side * padding_fraction / 100.0)
        else:
            raise ValueError(
                'Unknown padding size format: {}. Expected absolute values as "5px" or relative as "5%"'.format(
                    side_crop_config))
        if not do_crop:
            crop_pixels *= -1  # Pad instead of crop.
        return crop_pixels

    # TODO more informative error message.
    return Rectangle(top=get_crop_pixels(img_shape[0], TOP),
                     left=get_crop_pixels(img_shape[1], LEFT),
                     bottom=img_shape[0] - get_crop_pixels(img_shape[0], BOTTOM) - 1,
                     right=img_shape[1] - get_crop_pixels(img_shape[1], RIGHT) - 1)


def _make_cropped_rectangle(img_shape, crop_config):
    return _rectangle_from_cropping_or_padding_bounds(img_shape, crop_config, do_crop=True)


def _make_padded_rectangle(img_shape, pad_config):
    return _rectangle_from_cropping_or_padding_bounds(img_shape, pad_config, do_crop=False)


def _is_backend_only_geometry(geom: Geometry) -> bool:
    """Checks whether a figure is backend-only (i.e. should not be saved as a part of final image annotation)."""
    return any(isinstance(geom, geom_type) for geom_type in [MultichannelBitmap])


def _remove_backend_only_labels(labels):
    return [label for label in labels if not _is_backend_only_geometry(label.geometry)]


def _maybe_make_intermediate_bbox_class(save_config):
    return None if not save_config.get(SAVE, False) else ObjClass(name=save_config[CLASS_NAME],
                                                                  geometry_type=Rectangle)


def _maybe_make_bbox_label(roi: Rectangle, bbox_class: ObjClass, tags=None) -> list:
    return [Label(geometry=roi, obj_class=bbox_class, tags=tags)] if bbox_class is not None else []


def _get_annotation_for_bbox(img: np.ndarray, roi: Rectangle, model) -> Annotation:
    """Runs inference within the given roi; moves resulting figures to global reference frame."""
    img_cropped = roi.get_cropped_numpy_slice(img)
    # TODO pass through image and parent figure tags via roi_ann.
    roi_ann = Annotation(img_size=(roi.height, roi.width))
    raw_result_ann = model.inference(img_cropped, roi_ann)
    return Annotation(img_size=img.shape[:2],
                      labels=[label.translate(drow=roi.top, dcol=roi.left) for label in raw_result_ann.labels],
                      img_tags=raw_result_ann.img_tags, img_description=raw_result_ann.img_description,
                      pixelwise_scores_labels=[label.translate(drow=roi.top, dcol=roi.left)
                                               for label in raw_result_ann.pixelwise_scores_labels])


class InferenceModeBase:
    @staticmethod
    def mode_name():
        raise NotImplementedError()

    @classmethod
    def make_default_config(cls, model_result_suffix):
        return {
            MODEL_CLASSES: Renamer(add_suffix=model_result_suffix, enabled_classes=MATCH_ALL).to_json(),
            NAME: cls.mode_name()
        }

    def __init__(self, config: dict, in_meta: ProjectMeta, model: SingleImageInferenceBase):
        validation_schema_path = pkg_resources.resource_filename(
            __name__, 'inference_modes_schemas/{}.json'.format(self.mode_name()))
        MultiTypeValidator(validation_schema_path).val(INFERENCE_MODE_CONFIG, config)
        self._config = deepcopy(config)
        self._out_meta = in_meta.clone()
        self._model = model
        model_out_meta = self._model.model_out_meta
        renamer_model = Renamer.from_json(config[MODEL_CLASSES])
        for out_class in make_renamed_classes(model_out_meta.obj_classes, renamer_model, skip_missing=True):
            self._out_meta = self._out_meta.add_obj_class(out_class)
        self._model_class_mapper = RenamingObjClassMapper(dest_obj_classes=self._out_meta.obj_classes,
                                                          renamer=renamer_model)
        self._model_img_tags_renamer = Renamer(add_suffix=config[MODEL_CLASSES][Renamer.ADD_SUFFIX])
        self._renamed_model_img_tags = make_renamed_tag_metas(model_out_meta.img_tag_metas, self._model_img_tags_renamer,
                                                              skip_missing=True)
        self._model_img_tag_meta_mapper = RenamingTagMetaMapper(dest_tag_meta_dict=self._renamed_model_img_tags,
                                                                renamer=self._model_img_tags_renamer)
        self._out_meta = self._out_meta.merge(ProjectMeta(obj_tag_metas=model_out_meta.obj_tag_metas))

    @property
    def out_meta(self) -> ProjectMeta:
        return self._out_meta

    def infer_annotate(self, img: np.ndarray, ann: Annotation):
        result_ann = self._do_infer_annotate(img, ann)
        frontend_compatible_labels = _remove_backend_only_labels(result_ann.labels)
        return Annotation(img_size=result_ann.img_size, labels=frontend_compatible_labels, img_tags=result_ann.img_tags,
                          img_description=result_ann.img_description)

    def _do_infer_annotate(self, img, ann: Annotation) -> Annotation:
        raise NotImplementedError()


class InfModeFullImage(InferenceModeBase):
    @staticmethod
    def mode_name():
        return 'full_image'

    def __init__(self, config: dict, in_meta: ProjectMeta, model: SingleImageInferenceBase):
        super().__init__(config, in_meta, model)
        self._out_meta = self._out_meta.add_img_tag_metas(self._renamed_model_img_tags)

    def _do_infer_annotate(self, img: np.ndarray, ann: Annotation) -> Annotation:
        result_ann = ann.clone()
        inference_result_ann = self._model.inference(img, ann)
        result_ann = result_ann.add_labels(
            replace_labels_classes(inference_result_ann.labels, self._model_class_mapper, skip_missing=True))
        renamed_tags = make_renamed_tags(inference_result_ann.img_tags,
                                         self._model_img_tag_meta_mapper,
                                         skip_missing=True)
        result_ann = result_ann.add_tags(renamed_tags)
        return result_ann


class InfModeRoi(InferenceModeBase):
    @staticmethod
    def mode_name():
        return 'roi'

    @classmethod
    def make_default_config(cls, model_result_suffix: str):
        config = super(InfModeRoi, cls).make_default_config(model_result_suffix)
        our_config = {
            BOUNDS: {
                LEFT: '0' + PX,
                TOP: '0' + PX,
                RIGHT: '0' + PX,
                BOTTOM: '0' + PX,
            },
            SAVE: False,
            CLASS_NAME: 'inference_roi'
        }
        update_strict(config, our_config)
        return config

    def __init__(self, config: dict, in_meta: ProjectMeta, model: SingleImageInferenceBase):
        super().__init__(config, in_meta, model)
        self._out_meta = self._out_meta.add_img_tag_metas(self._renamed_model_img_tags)

        self._intermediate_bbox_class = _maybe_make_intermediate_bbox_class(self._config)
        if self._intermediate_bbox_class is not None:
            self._out_meta = self._out_meta.add_obj_class(self._intermediate_bbox_class)
            self._out_meta = self._out_meta.add_obj_tag_metas(self._model.model_out_meta.img_tag_metas)

    def _do_infer_annotate(self, img: np.ndarray, ann: Annotation) -> Annotation:
        result_ann = ann.clone()
        roi = _make_cropped_rectangle(ann.img_size, self._config[BOUNDS])
        roi_ann = _get_annotation_for_bbox(img, roi, self._model)
        result_ann = result_ann.add_labels(
            replace_labels_classes(roi_ann.labels, self._model_class_mapper, skip_missing=True))
        img_level_tags = make_renamed_tags(roi_ann.img_tags, self._model_img_tag_meta_mapper, skip_missing=True)
        result_ann = result_ann.add_labels(
            _maybe_make_bbox_label(roi, self._intermediate_bbox_class, tags=roi_ann.img_tags))
        result_ann = result_ann.add_tags(img_level_tags)
        return result_ann


class InfModeBboxes(InferenceModeBase):
    @staticmethod
    def mode_name():
        return 'bboxes'

    @classmethod
    def make_default_config(cls, model_result_suffix: str) -> dict:
        config = super(InfModeBboxes, cls).make_default_config(model_result_suffix)
        our_config = {
            FROM_CLASSES: MATCH_ALL,
            PADDING: {
                LEFT: '0' + PX,
                TOP: '0' + PX,
                RIGHT: '0' + PX,
                BOTTOM: '0' + PX,
            },
            SAVE: False,
            Renamer.ADD_SUFFIX: '_input_bbox'
        }
        update_strict(config, our_config)
        return config

    def __init__(self, config: dict, in_meta: ProjectMeta, model: SingleImageInferenceBase):
        super().__init__(config, in_meta, model)

        self._renamer_intermediate = None
        if self._config[SAVE]:
            renamer_intermediate = Renamer(add_suffix=self._config.get(Renamer.ADD_SUFFIX, ''),
                                           enabled_classes=self._config[FROM_CLASSES])
            intermediate_renamed_classes = make_renamed_classes(in_meta.obj_classes, renamer_intermediate,
                                                                skip_missing=True)
            for renamed_class in intermediate_renamed_classes:
                self._out_meta = self._out_meta.add_obj_class(renamed_class.clone(geometry_type=Rectangle))
            self._intermediate_class_mapper = RenamingObjClassMapper(dest_obj_classes=self._out_meta.obj_classes,
                                                                     renamer=renamer_intermediate)
            self._out_meta = self._out_meta.add_obj_tag_metas(self._renamed_model_img_tags)

    def _do_infer_annotate(self, img: np.ndarray, ann: Annotation) -> Annotation:
        result_labels = []
        for src_label, roi in self._all_filtered_bbox_rois(ann, self._config[FROM_CLASSES], self._config[PADDING]):
            if roi is None:
                result_labels.append(src_label)
            else:
                roi_ann = _get_annotation_for_bbox(img, roi, self._model)
                result_labels.extend(replace_labels_classes(
                    roi_ann.labels, self._model_class_mapper, skip_missing=True))
                model_img_level_tags = make_renamed_tags(roi_ann.img_tags, self._model_img_tag_meta_mapper,
                                                         skip_missing=True)
                if self._config[SAVE]:
                    result_labels.append(
                        Label(geometry=roi, obj_class=self._intermediate_class_mapper.map(src_label.obj_class),
                              tags=model_img_level_tags))
                result_labels.append(src_label.add_tags(model_img_level_tags))
        return ann.clone(labels=result_labels)

    @staticmethod
    def _all_filtered_bbox_rois(ann: Annotation, included_classes, crop_config: dict):
        for src_label in ann.labels:
            effective_roi = None
            if is_name_included(src_label.obj_class.name, included_classes):
                bbox = src_label.geometry.to_bbox()
                roi = _make_padded_rectangle((bbox.height, bbox.width), crop_config)
                maybe_effective_roi = roi.translate(drow=bbox.top, dcol=bbox.left).crop(
                    Rectangle.from_size(ann.img_size))
                if len(maybe_effective_roi) > 0:
                    [effective_roi] = maybe_effective_roi
            yield src_label, effective_roi


class InfModeSlidinglWindowBase(InferenceModeBase):
    def __init__(self, config: dict, in_meta: ProjectMeta, model: SingleImageInferenceBase):
        super().__init__(config, in_meta, model)

        window_shape = (self._config[WINDOW][HEIGHT], self._config[WINDOW][WIDTH])
        min_overlap = (self._config[MIN_OVERLAP][Y], self._config[MIN_OVERLAP][X])
        self._sliding_windows = SlidingWindows(window_shape, min_overlap)

        self._intermediate_bbox_class = _maybe_make_intermediate_bbox_class(self._config)
        if self._intermediate_bbox_class is not None:
            self._out_meta = self._out_meta.add_obj_class(self._intermediate_bbox_class)
            self._out_meta = self._out_meta.add_obj_tag_metas(self._renamed_model_img_tags)

    @classmethod
    def make_default_config(cls, model_result_suffix: str) -> dict:
        config = super(InfModeSlidinglWindowBase, cls).make_default_config(model_result_suffix)
        our_config = {
            WINDOW: {
                WIDTH: 128,
                HEIGHT: 128,
            },
            MIN_OVERLAP: {
                X: 0,
                Y: 0,
            },
            SAVE: False,
            CLASS_NAME: 'sliding_window',
        }
        update_strict(config, our_config)
        return config


# This only makes sense for image segmentation that return per-pixel class probabilities.
class InfModeSlidingWindowSegmentation(InfModeSlidinglWindowBase):
    @staticmethod
    def mode_name():
        return 'sliding_window'

    def _do_infer_annotate(self, img: np.ndarray, ann: Annotation) -> Annotation:
        result_ann = ann.clone()
        all_pixelwise_scores_labels = []
        for roi in self._sliding_windows.get(ann.img_size):
            raw_roi_ann = _get_annotation_for_bbox(img, roi, self._model)
            all_pixelwise_scores_labels.extend(raw_roi_ann.pixelwise_scores_labels)
            model_img_level_tags = make_renamed_tags(raw_roi_ann.img_tags, self._model_img_tag_meta_mapper,
                                                     make_renamed_tags)
            result_ann = result_ann.add_labels(
                _maybe_make_bbox_label(roi, self._intermediate_bbox_class, tags=model_img_level_tags))
        model_class_name_to_id = {name: idx
                                  for idx, name in enumerate(set(label.obj_class.name
                                                                 for label in all_pixelwise_scores_labels))}
        id_to_class_obj = {idx: self._model.model_out_meta.obj_classes.get(name)
                           for name, idx in model_class_name_to_id.items()}
        summed_scores = np.zeros(ann.img_size + tuple([len(model_class_name_to_id)]))
        for label in all_pixelwise_scores_labels:
            class_idx = model_class_name_to_id[label.obj_class.name]
            label_matching_summer_scores = label.geometry.to_bbox().get_cropped_numpy_slice(summed_scores)
            label_matching_summer_scores[:, :, class_idx, np.newaxis] += label.geometry.data
        # TODO consider instead filtering pixels by all-zero scores.
        if np.sum(summed_scores, axis=2).min() == 0:
            raise RuntimeError('Wrong sliding window moving, implementation error.')
        aggregated_model_labels = raw_to_labels.segmentation_array_to_sly_bitmaps(id_to_class_obj,
                                                                                  np.argmax(summed_scores, axis=2))
        result_ann = result_ann.add_labels(
            replace_labels_classes(aggregated_model_labels, self._model_class_mapper, skip_missing=True))
        return result_ann


class InfModeSlidingWindowDetection(InfModeSlidinglWindowBase):
    @staticmethod
    def mode_name():
        return 'sliding_window_det'

    @classmethod
    def make_default_config(cls, model_result_suffix: str) -> dict:
        config = super(InfModeSlidingWindowDetection, cls).make_default_config(model_result_suffix)
        our_config = {
            NMS_AFTER: {
                ENABLE: False,
                IOU_THRESHOLD: 0.0,
                CONFIDENCE_TAG_NAME: CONFIDENCE
            },
        }
        update_strict(config, our_config)
        return config

    @staticmethod
    def _iou(first_rect, second_rect):
        intersection_rects = first_rect.crop(second_rect)
        if len(intersection_rects) == 0:
            return 0
        intersection_rect = intersection_rects[0]
        return safe_ratio(intersection_rect.area, first_rect.area + second_rect.area - intersection_rect.area)

    @classmethod
    def _single_class_nms(cls, labels, iou_thresh, confidence_tag_name):
        sorted_labels = sorted(labels, key=lambda x: -x.tags.get(confidence_tag_name).value)
        result_labels = []
        while len(sorted_labels) > 0:
            result_labels.append(sorted_labels.pop(0))
            sorted_labels = [label for label in sorted_labels
                             if cls._iou(label.geometry.to_bbox(), result_labels[-1].geometry.to_bbox()) <= iou_thresh]
        return result_labels

    # @TODO: move out
    @classmethod
    def _general_nms(cls, labels, iou_thresh, confidence_tag_name):
        if not all(isinstance(label.geometry, Rectangle) for label in labels):
            raise RuntimeError('Non-max suppression expects labels with Rectangle geometry.')
        if not all(label.tags.has_key(confidence_tag_name) for label in labels):
            raise RuntimeError('Non-max suppression expects "{}" tag in labels.'.format(confidence_tag_name))
        if not all(label.tags.get(confidence_tag_name).meta.value_type == TagValueType.ANY_NUMBER for label in labels):
            raise RuntimeError('Non-max suppression expects "{}" tag to have type {}.'.format(confidence_tag_name,
                                                                                              TagValueType.ANY_NUMBER))

        # Group labels by their respective class.
        labels_per_class = defaultdict(list)
        for label in labels:
            labels_per_class[label.obj_class.name].append(label)
        survived_labels = [cls._single_class_nms(single_class_labels, iou_thresh, confidence_tag_name) for
                           single_class_labels in labels_per_class.values()]
        return list(itertools.chain(*survived_labels))

    def _do_infer_annotate(self, img: np.ndarray, ann: Annotation) -> Annotation:
        result_ann = ann.clone()
        model_labels = []
        for roi in self._sliding_windows.get(ann.img_size):
            raw_roi_ann = _get_annotation_for_bbox(img, roi, self._model)
            all_rectangle_labels = [label for label in raw_roi_ann.labels if isinstance(label.geometry, Rectangle)]
            model_labels.extend(
                replace_labels_classes(all_rectangle_labels, self._model_class_mapper, skip_missing=True))
            model_img_level_tags = make_renamed_tags(raw_roi_ann.img_tags, self._model_img_tag_meta_mapper,
                                                     skip_missing=True)
            result_ann = result_ann.add_labels(_maybe_make_bbox_label(roi, self._intermediate_bbox_class, tags=model_img_level_tags))

        nms_conf = self._config.get(NMS_AFTER, {ENABLE: False})
        if nms_conf[ENABLE]:
            result_ann = result_ann.add_labels(self._general_nms(labels=model_labels, iou_thresh=nms_conf[IOU_THRESHOLD],
                                                    confidence_tag_name=nms_conf.get(CONFIDENCE_TAG_NAME, CONFIDENCE)))
        else:
            result_ann = result_ann.add_labels(model_labels)
        return result_ann


class InferenceModeFactory:
    mapping = {inference_mode_cls.mode_name(): inference_mode_cls
               for inference_mode_cls in [InfModeFullImage,
                                          InfModeRoi,
                                          InfModeBboxes,
                                          InfModeSlidingWindowSegmentation,
                                          InfModeSlidingWindowDetection]}

    @classmethod
    def create(cls, config, *args, **kwargs):
        key = config[NAME]
        feeder_cls = cls.mapping.get(key)
        if feeder_cls is None:
            raise NotImplementedError()
        res = feeder_cls(config, *args, **kwargs)
        return res


def get_effective_inference_mode_config(task_inference_mode_config: dict, default_inference_mode_config: dict) -> dict:
    task_inference_mode_name = task_inference_mode_config.get(NAME, None)
    default_inference_mode_name = default_inference_mode_config.get(NAME, None)
    inference_mode_name = task_inference_mode_name or default_inference_mode_name
    if inference_mode_name is None:
        raise RuntimeError(
            'Inference mode name ({} key) not found in mode config.'.format(NAME))

    inference_mode_cls = InferenceModeFactory.mapping[inference_mode_name]

    # Priorities when filling out inference mode config:
    # - task config (highest)
    # - default config from constructor
    # - default config for the given mode (lowest).
    result_config = inference_mode_cls.make_default_config(model_result_suffix='')
    if default_inference_mode_name == inference_mode_name:
        # Only take custom mode defaults into account if the mode name matches the task config.
        update_recursively(result_config, default_inference_mode_config)
    update_recursively(result_config, task_inference_mode_config)
    return result_config
