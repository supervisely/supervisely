# coding: utf-8

from supervisely_lib.project.project import Project
from supervisely_lib.annotation.tag_meta_collection import TagMetaCollection
from supervisely_lib.annotation.tag_meta import TagValueType
from supervisely_lib.sly_logger import logger


CLASSES_MAPPING = 'classes_mapping'
TAGS_MAPPING = 'tags_mapping'

TRUE_POSITIVE = 'true-positive'
TRUE_NEGATIVE = 'true-negative'
FALSE_POSITIVE = 'false-positive'
FALSE_NEGATIVE = 'false-negative'

ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1_MEASURE = 'F1-measure'
TOTAL = 'total'
TOTAL_GROUND_TRUTH = 'total-ground-truth'
TOTAL_PREDICTIONS = 'total-predictions'


def check_class_mapping(first_project: Project, second_project: Project, classes_mapping: dict) -> None:
    for k, v in classes_mapping.items():
        if first_project.meta.obj_classes.get(k) is None:
            raise RuntimeError('Class {} does not exist in input project "{}".'.format(k, first_project.name))
        if second_project.meta.obj_classes.get(v) is None:
            raise RuntimeError('Class {} does not exist in input project "{}".'.format(v, second_project.name))


def _get_no_value_tags(tag_metas: TagMetaCollection) -> list:
    return [tag_meta.name for tag_meta in tag_metas if tag_meta.value_type == TagValueType.NONE]


def check_tag_mapping(first_project: Project, second_project: Project, tags_mapping: dict) -> None:
    project_first_tags = _get_no_value_tags(first_project.meta.img_tag_metas)
    project_second_tags = _get_no_value_tags(second_project.meta.img_tag_metas)

    for k, v in tags_mapping.items():
        if k not in project_first_tags:
            raise RuntimeError('Tag {} does not exist in input project "{}".'.format(k, first_project.name))
        if v not in project_second_tags:
            raise RuntimeError('Tag {} does not exist in input project "{}".'.format(v, second_project.name))


def safe_ratio(num, denom):
    return (num / denom) if denom != 0 else 0


def sum_counters(elementwise_counters, counter_names):
    return {counter_name: sum(c.get(counter_name, 0) for c in elementwise_counters) for counter_name in counter_names}


def log_line(length=80, c=' '):
    logger.info(c * length)


def log_head(string):
    logger.info(string.center(80, '*'))
