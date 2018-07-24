# coding: utf-8

from collections import defaultdict
from threading import Lock

import numpy as np

from ..project import Annotation
from ..figure import FigClasses, FigureBitmap, FigureRectangle, Rect
from ..sly_logger import logger
from .json_utils import json_load


def samples_by_tags(tags, project_fs, project_meta):
    samples = defaultdict(list)
    for item_descr in project_fs:
        ann_packed = json_load(item_descr.ann_path)
        ann = Annotation.from_packed(ann_packed, project_meta)
        for req_tag in tags:
            if (req_tag == '__all__') or (req_tag in ann['tags']):
                samples[req_tag].append(item_descr)

    return samples


class CorruptedSampleCatcher(object):
    def __init__(self, allow_corrupted_cnt):
        self.fails_allowed = allow_corrupted_cnt
        self._failed_uids = set()
        self._lock = Lock()

    def exec(self, uid, log_dct, f, *args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            self._lock.acquire()
            if uid not in self._failed_uids:
                self._failed_uids.add(uid)
                logger.warn('Sample processing error.', extra=log_dct, exc_info=True)
            fail_cnt = len(self._failed_uids)
            self._lock.release()

            if fail_cnt > self.fails_allowed:
                raise RuntimeError('Too many errors occured while processing samples. '
                                   'Allowed: {}.'.format(self.fails_allowed))


# enumerates input classes and produces color mapping (default rule) and model out classes
def create_segmentation_classes(in_project_classes,
                                bkg_title, neutral_title,
                                bkg_color, neutral_color):
    in_project_titles = sorted((x['title'] for x in in_project_classes))

    # determine mapping to color (idx)
    class_title_to_idx = {
        bkg_title: bkg_color,
        neutral_title: neutral_color
    }
    tmp_titles = filter(lambda x: x != bkg_title and x != neutral_title, in_project_titles)
    for i, title in enumerate(tmp_titles):
        class_title_to_idx[title] = i + 1  # usually bkg_color is 0

    if len(set(class_title_to_idx.values())) != len(class_title_to_idx):
        raise RuntimeError('Unable to construct internal color mapping for classes.')

    # determine out classes
    out_classes = FigClasses()
    if bkg_title not in in_project_titles:
        out_classes.add({
            'title': bkg_title,
            'shape': 'bitmap',
            'color': '#222222',  # @TODO: add shared_utils.gen_new_color() analogue for rand color
        })  # add bkg class to out if not defined

    for in_class in in_project_classes:
        title = in_class['title']
        if title == neutral_title:
            continue  # exclude neutral from out
        out_classes.add({
            'title': title,
            'shape': 'bitmap',
            'color': in_class['color'],
        })

    return class_title_to_idx, out_classes


def create_detection_classes(in_project_classes):
    in_project_titles = sorted((x['title'] for x in in_project_classes))

    class_title_to_idx = {}
    for i, title in enumerate(in_project_titles):
        class_title_to_idx[title] = i + 1  # usually bkg_color is 0

    if len(set(class_title_to_idx.values())) != len(class_title_to_idx):
        raise RuntimeError('Unable to construct internal color mapping for classes.')

    # determine out classes
    out_classes = FigClasses()

    for in_class in in_project_classes:
        title = in_class['title']
        out_classes.add({
            'title': title,
            'shape': 'rectangle',
            'color': in_class['color'],
        })

    return class_title_to_idx, out_classes


# converts predictions (encoded as numbers)
def prediction_to_sly_bitmaps(class_title_to_idx, pred):
    size_wh = (pred.shape[1], pred.shape[0])
    out_figures = []
    for cls_title in sorted(class_title_to_idx.keys()):
        cls_idx = class_title_to_idx[cls_title]
        class_pred_mask = pred == cls_idx
        new_objs = FigureBitmap.from_mask(cls_title, size_wh, origin=(0, 0), mask=class_pred_mask)
        out_figures.extend(new_objs)
    return out_figures


# converts tf_model inference output
def detection_preds_to_sly_rects(inverse_mapping, net_out, img_shape, min_score_thresold):
    img_wh = img_shape[1::-1]
    (boxes, scores, classes, num) = net_out
    out_figures = []
    thr_mask = np.squeeze(scores) > min_score_thresold
    for box, class_id, score in zip(np.squeeze(boxes)[thr_mask],
                                    np.squeeze(classes)[thr_mask],
                                    np.squeeze(scores)[thr_mask]):

            xmin = int(box[1] * img_shape[1])
            ymin = int(box[0] * img_shape[0])
            xmax = int(box[3] * img_shape[1])
            ymax = int(box[2] * img_shape[0])
            cls_name = inverse_mapping[int(class_id)]
            rect = Rect(xmin, ymin, xmax, ymax)
            new_objs = FigureRectangle.from_rect(cls_name, img_wh, rect)
            for x in new_objs:
                x.data['score'] = float(score)
            out_figures.extend(new_objs)
    return out_figures
