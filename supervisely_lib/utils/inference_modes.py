# coding: utf-8

from copy import deepcopy
from enum import Enum

import numpy as np

from ..figure.fig_classes import FigClasses
from ..figure.rectangle import Rect
from ..figure.figure_rectangle import FigureRectangle
from ..figure.sliding_windows import SlidingWindows
from ..figure.aux import crop_image_with_rect
from .config_readers import rect_from_bounds
from .nn_data import prediction_to_sly_bitmaps


class ObjRenamer:
    def __init__(self, given_classes, rules, forced_shape=None):
        self.suffix = rules['add_suffix']
        self.en_class_titles = rules['save_classes']

        self.out_mapping = {}
        self.out_classes = FigClasses()
        for src_cls in given_classes:
            src_title = src_cls['title']
            if (self.en_class_titles != '__all__') and (src_title not in self.en_class_titles):
                continue
            new_title = src_title + self.suffix
            self.out_mapping[src_title] = new_title
            new_shape = src_cls['shape'] if forced_shape is None else forced_shape
            self.out_classes.add({**deepcopy(src_cls), 'title': new_title, 'shape': new_shape})

        self.allowed_titles = set(self.out_mapping.keys())

    def process_objs(self, objs):
        objs = [x for x in objs if x.class_title in self.allowed_titles]
        for fig in objs:
            fig.class_title = self.out_mapping[fig.class_title]
        return objs


class InfResultsToFeeder(Enum):
    FIGURES = 1  # expects sly figures as a result
    SEGMENTATION = 2  # expects class mapping & probabilities as a result


class InfFeederFullImage:
    expected_result = InfResultsToFeeder.FIGURES

    def __init__(self, settings, in_pr_meta, model_out_classes):
        self._renamer_existing = ObjRenamer(in_pr_meta.classes, settings['existing_objects'])
        self._renamer_found = ObjRenamer(model_out_classes, settings['model_classes'])

        names_0 = self._renamer_existing.out_classes.unique_names
        names_1 = self._renamer_found.out_classes.unique_names
        if len(names_0 & names_1) > 0:
            raise RuntimeError('Unable to determine output classes due to non-unique result class names.')

        self._out_meta = deepcopy(in_pr_meta)
        out_classes = FigClasses()
        out_classes.update(self._renamer_existing.out_classes)
        out_classes.update(self._renamer_found.out_classes)
        self._out_meta.classes = out_classes

    @property
    def out_meta(self):
        return self._out_meta

    def feed(self, img, ann, inference_cback):
        found_objects = inference_cback(img, ann)
        src_objects = self._renamer_existing.process_objs(ann['objects'])
        found_objects = self._renamer_found.process_objs(found_objects)
        ann['objects'] = src_objects + found_objects
        return ann


class InfFeederRoi:
    expected_result = InfResultsToFeeder.FIGURES

    def __init__(self, settings, in_pr_meta, model_out_classes):
        self.mode_conf = settings['mode']

        self._renamer_existing = ObjRenamer(in_pr_meta.classes, settings['existing_objects'])
        self._renamer_found = ObjRenamer(model_out_classes, settings['model_classes'])
        self._interm_classes = FigClasses()
        if self.mode_conf['save']:
            self._interm_classes.add({
                'title': self.mode_conf['class_name'],
                'shape': 'rectangle'
            })  # random color

        names_0 = self._renamer_existing.out_classes.unique_names
        names_1 = self._renamer_found.out_classes.unique_names
        names_2 = self._interm_classes.unique_names
        if len(names_0 | names_1 | names_2) != len(names_0) + len(names_1) + len(names_2):
            raise RuntimeError('Unable to determine output classes due to non-unique result class names.')

        self._out_meta = deepcopy(in_pr_meta)
        out_classes = FigClasses()
        for x in (self._renamer_existing.out_classes, self._renamer_found.out_classes, self._interm_classes):
            out_classes.update(x)
        self._out_meta.classes = out_classes

    @property
    def out_meta(self):
        return self._out_meta

    def feed(self, img, ann, inference_cback):
        img_wh = ann.image_size_wh
        roi = rect_from_bounds(self.mode_conf['bounds'], *img_wh)
        rect_img = Rect.from_size(img_wh)
        if roi.is_empty or (not rect_img.contains(roi)):
            raise RuntimeError('Mode "roi": result crop bounds are invalid.')

        img_cropped = crop_image_with_rect(img, roi)
        found_objects = inference_cback(img_cropped, None)  # no need to crop & pass figures now
        for fig in found_objects:
            fig.shift((roi.left, roi.top))  # and no need to normalize

        src_objects = self._renamer_existing.process_objs(ann['objects'])
        found_objects = self._renamer_found.process_objs(found_objects)
        interm_objects = []
        if self.mode_conf['save']:
            interm_objects.extend(FigureRectangle.from_rect(self.mode_conf['class_name'], img_wh, roi))

        ann['objects'] = src_objects + interm_objects + found_objects
        return ann


class InfFeederBboxes:
    expected_result = InfResultsToFeeder.FIGURES

    def __init__(self, settings, in_pr_meta, model_out_classes):
        self.mode_conf = settings['mode']

        self._renamer_existing = ObjRenamer(in_pr_meta.classes, settings['existing_objects'])
        self._renamer_found = ObjRenamer(model_out_classes, settings['model_classes'])

        bbox_in_classes = in_pr_meta.classes if self.mode_conf['save'] else FigClasses()
        self._renamer_bboxes = ObjRenamer(bbox_in_classes, forced_shape='rectangle', rules={
            'add_suffix': self.mode_conf['add_suffix'],
            'save_classes': self.mode_conf['from_classes']
        })

        names_0 = self._renamer_existing.out_classes.unique_names
        names_1 = self._renamer_found.out_classes.unique_names
        names_2 = self._renamer_bboxes.out_classes.unique_names
        if len(names_0 | names_1 | names_2) != len(names_0) + len(names_1) + len(names_2):
            raise RuntimeError('Unable to determine output classes due to non-unique result class names.')

        self._out_meta = deepcopy(in_pr_meta)
        out_classes = FigClasses()
        for x in (self._renamer_existing, self._renamer_found, self._renamer_bboxes):
            out_classes.update(x.out_classes)
        self._out_meta.classes = out_classes

    @property
    def out_meta(self):
        return self._out_meta

    def feed(self, img, ann, inference_cback):
        img_wh = ann.image_size_wh
        rect_img = Rect.from_size(img_wh)
        src_titles = self.mode_conf['from_classes']

        found_objects = []
        interm_objects = []
        for src_obj in ann['objects']:
            if (src_titles != '__all__') and (src_obj.class_title not in src_titles):
                continue
            bbox = src_obj.get_bbox().round()  # ok, now we are working with integer rects
            roi = rect_from_bounds(self.mode_conf['padding'],
                                   img_w=bbox.width, img_h=bbox.height, shift_inside=False)
            roi = roi.move((bbox.left, bbox.top))
            if roi.is_empty:
                continue
            roi = roi.intersection(rect_img)
            if roi.is_empty:
                continue

            img_cropped = crop_image_with_rect(img, roi)

            found_from_bbox = inference_cback(img_cropped, None)  # no need to crop & pass figures now
            for fig in found_from_bbox:
                fig.shift((roi.left, roi.top))  # and no need to normalize
            found_objects.extend(found_from_bbox)

            if self.mode_conf['save']:
                interm_objects.extend(FigureRectangle.from_rect(src_obj.class_title, img_wh, roi))

        src_objects = self._renamer_existing.process_objs(ann['objects'])
        found_objects = self._renamer_found.process_objs(found_objects)
        interm_objects = self._renamer_bboxes.process_objs(interm_objects)
        ann['objects'] = src_objects + interm_objects + found_objects
        return ann


# for image segmentation only
class InfFeederSlWindow:
    expected_result = InfResultsToFeeder.SEGMENTATION

    def __init__(self, settings, in_pr_meta, model_out_classes):
        self.class_cnt = len(model_out_classes)
        self.mode_conf = settings['mode']
        window_wh = (self.mode_conf['window']['width'], self.mode_conf['window']['height'])
        min_overlap_xy = (self.mode_conf['min_overlap']['x'], self.mode_conf['min_overlap']['y'])
        self.sliding_windows = SlidingWindows(window_wh, min_overlap_xy)  # + some validating

        self._renamer_existing = ObjRenamer(in_pr_meta.classes, settings['existing_objects'])
        self._renamer_found = ObjRenamer(model_out_classes, settings['model_classes'])
        self._interm_classes = FigClasses()
        if self.mode_conf['save']:
            self._interm_classes.add({
                'title': self.mode_conf['class_name'],
                'shape': 'rectangle'
            })  # random color

        names_0 = self._renamer_existing.out_classes.unique_names
        names_1 = self._renamer_found.out_classes.unique_names
        names_2 = self._interm_classes.unique_names
        if len(names_0 | names_1 | names_2) != len(names_0) + len(names_1) + len(names_2):
            raise RuntimeError('Unable to determine output classes due to non-unique result class names.')

        self._out_meta = deepcopy(in_pr_meta)
        out_classes = FigClasses()
        for x in (self._renamer_existing.out_classes, self._renamer_found.out_classes, self._interm_classes):
            out_classes.update(x)
        self._out_meta.classes = out_classes

    @property
    def out_meta(self):
        return self._out_meta

    def feed(self, img, ann, inference_cback):
        img_wh = ann.image_size_wh
        interm_objects = []
        buffer = np.zeros((img_wh[1], img_wh[0], self.class_cnt), dtype=np.float64)
        cnt_buffer = np.zeros((img_wh[1], img_wh[0]), dtype=np.int32)
        cls_mapping = None
        for roi in self.sliding_windows.get(img_wh):
            img_cropped = crop_image_with_rect(img, roi)
            cls_mapping, pred_window = inference_cback(img_cropped, None)

            buffer[roi.top:roi.bottom, roi.left:roi.right, :] += pred_window
            cnt_buffer[roi.top:roi.bottom, roi.left:roi.right] += 1

            if self.mode_conf['save']:
                new_rect = FigureRectangle.from_rect(self.mode_conf['class_name'], img_wh, roi)
                interm_objects.extend(new_rect)

        if cnt_buffer.min() < 1:
            raise RuntimeError('Wrong sliding window moving, implementation error.')
        cnt_buffer = np.repeat(cnt_buffer[:, :, np.newaxis], self.class_cnt, axis=2)
        buffer = buffer / cnt_buffer
        out_pred = np.argmax(buffer, axis=2)
        found_objects = prediction_to_sly_bitmaps(cls_mapping, out_pred)

        src_objects = self._renamer_existing.process_objs(ann['objects'])
        found_objects = self._renamer_found.process_objs(found_objects)
        ann['objects'] = src_objects + interm_objects + found_objects
        return ann


class InfFeederSlWindowDetection:
    expected_result = InfResultsToFeeder.FIGURES

    def __init__(self, settings, in_pr_meta, model_out_classes):
        self.class_cnt = len(model_out_classes)
        self.mode_conf = settings['mode']
        window_wh = (self.mode_conf['window']['width'], self.mode_conf['window']['height'])
        min_overlap_xy = (self.mode_conf['min_overlap']['x'], self.mode_conf['min_overlap']['y'])
        self.sliding_windows = SlidingWindows(window_wh, min_overlap_xy)  # + some validating

        self._renamer_existing = ObjRenamer(in_pr_meta.classes, settings['existing_objects'])
        self._renamer_found = ObjRenamer(model_out_classes, settings['model_classes'])
        self._interm_classes = FigClasses()
        if self.mode_conf['save']:
            self._interm_classes.add({
                'title': self.mode_conf['class_name'],
                'shape': 'rectangle'
            })  # random color

        names_0 = self._renamer_existing.out_classes.unique_names
        names_1 = self._renamer_found.out_classes.unique_names
        names_2 = self._interm_classes.unique_names
        if len(names_0 | names_1 | names_2) != len(names_0) + len(names_1) + len(names_2):
            raise RuntimeError('Unable to determine output classes due to non-unique result class names.')

        self._out_meta = deepcopy(in_pr_meta)
        out_classes = FigClasses()
        for x in (self._renamer_existing.out_classes, self._renamer_found.out_classes, self._interm_classes):
            out_classes.update(x)
        self._out_meta.classes = out_classes

    # 'max' NMS
    @classmethod
    def _single_class_nms(cls, figures_rect, iou_thresh):
        incr_score = sorted(figures_rect, key=lambda x: x.data['score'])  # ascending
        out_figs = []
        for curr_fig in incr_score:
            curr_bbox = curr_fig.get_bbox()
            # suppress earlier (with less thresh)
            out_figs = list(filter(lambda x: x.get_bbox().iou(curr_bbox) <= iou_thresh, out_figs))
            out_figs.append(curr_fig)

        return out_figs

    # @TODO: move out
    @classmethod
    def general_nms(cls, figures_rect, iou_thresh):
        if not all(isinstance(x, FigureRectangle) for x in figures_rect):
            raise RuntimeError('NMS expects FigureRectangle.')
        if not all('score' in x.data for x in figures_rect):
            raise RuntimeError('NMS expects "score" field in figures.')

        use_classes = set(x.class_title for x in figures_rect)
        res = []
        for cls_title in sorted(list(use_classes)):
            class_figures = list(filter(lambda x: x.class_title == cls_title, figures_rect))
            res.extend(cls._single_class_nms(class_figures, iou_thresh))
        return res

    @property
    def out_meta(self):
        return self._out_meta

    def feed(self, img, ann, inference_cback):
        nms_conf = self.mode_conf['nms_after']
        img_wh = ann.image_size_wh
        interm_objects = []
        found_objects = []
        for roi in self.sliding_windows.get(img_wh):
            img_cropped = crop_image_with_rect(img, roi)
            figures_from_window = inference_cback(img_cropped, None)
            for fig in figures_from_window:
                fig.shift((roi.left, roi.top))  # and no need to normalize
            found_objects.extend(figures_from_window)

            if self.mode_conf['save']:
                new_rect = FigureRectangle.from_rect(self.mode_conf['class_name'], img_wh, roi)
                interm_objects.extend(new_rect)

        if nms_conf['enable']:
            found_objects = self.general_nms(figures_rect=found_objects, iou_thresh=nms_conf['iou_threshold'])

        src_objects = self._renamer_existing.process_objs(ann['objects'])
        found_objects = self._renamer_found.process_objs(found_objects)
        ann['objects'] = src_objects + interm_objects + found_objects
        return ann


class InferenceFeederFactory:
    mapping = {
        'full_image': InfFeederFullImage,
        'roi': InfFeederRoi,
        'bboxes': InfFeederBboxes,
        'sliding_window': InfFeederSlWindow,
        'sliding_window_det': InfFeederSlWindowDetection,
    }

    @classmethod
    def create(cls, settings, *args, **kwargs):
        key = settings['mode']['source']
        feeder_cls = cls.mapping.get(key)
        if feeder_cls is None:
            raise NotImplemented()
        res = feeder_cls(settings, *args, **kwargs)
        return res
