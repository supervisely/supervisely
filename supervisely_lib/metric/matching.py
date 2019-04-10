# coding: utf-8
from collections import namedtuple

from supervisely_lib.annotation.label import Label
from supervisely_lib.metric.common import safe_ratio

import numpy as np


LabelsPairWithScore = namedtuple('LabelsPairWithScore', ['label_1', 'label_2', 'score'])
LabelsMatchResult = namedtuple('LabelsMatchResult', ['matches', 'unmatched_labels_1', 'unmatched_labels_2'])


def filter_labels_by_name(labels, names_whitelist):
    return [label for label in labels if label.obj_class.name in names_whitelist]


def get_labels_iou(label_1: Label, label_2: Label, img_size):
    mask_1 = np.full(img_size, 0, dtype=np.uint8)
    label_1.geometry.draw(mask_1, color=1)
    mask_2 = np.full(img_size, 0, dtype=np.uint8)
    label_2.geometry.draw(mask_2, color=1)
    return safe_ratio((mask_1 & mask_2).sum(), (mask_1 | mask_2).sum())


def match_labels_by_iou(labels_1, labels_2, img_size, iou_threshold):
    # Score all the possible label pairs.
    scored_label_pairs = [
        LabelsPairWithScore(label_1=label_1, label_2=label_2, score=get_labels_iou(label_1, label_2, img_size))
        for label_1 in labels_1 for label_2 in labels_2]
    # Apply the threshold to avoid sorting candidates with too low scores.
    thresholded_label_pairs = [p for p in scored_label_pairs if p.score >= iou_threshold]
    # Sort by score in descending order.
    sorted_label_pairs = sorted(thresholded_label_pairs, key=lambda p: p.score, reverse=True)

    # Match greedily, make sure no label is matched to more than one counterpart.
    unmatched_labels_1 = set(labels_1)
    unmatched_labels_2 = set(labels_2)
    matches = []
    for label_pair in sorted_label_pairs:
        if label_pair.label_1 in unmatched_labels_1 and label_pair.label_2 in unmatched_labels_2:
            matches.append(label_pair)
            unmatched_labels_1.remove(label_pair.label_1)
            unmatched_labels_2.remove(label_pair.label_2)
    return LabelsMatchResult(matches=matches, unmatched_labels_1=unmatched_labels_1,
                             unmatched_labels_2=unmatched_labels_2)
