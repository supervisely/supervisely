import albumentations as A
from supervisely_lib.sly_logger import logger
from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.geometry.bitmap import Bitmap
from supervisely_lib.geometry.rectangle import Rectangle
from supervisely_lib.geometry.polygon import Polygon
import numpy as np


prob_already_add = 'p='
unsupported_bbox_albs = [A.CoarseDropout, A.ElasticTransform, A.OpticalDistortion, A.RandomGridShuffle,
                         A.GridDistortion, A.GridDropout, A.MaskDropout]
unsupport_bbox = False


def get_coco_bbox(rect):
    return [rect.left, rect.top, rect.width, rect.height]


def create_albumentations_info(category_name, aug_name, params, sometimes=None):
    clean_params = params
    res = {
        "category": category_name,
        "name": aug_name,
        "params": clean_params,
    }
    if sometimes is not None:
        if not (0.0 <= sometimes <= 1):
            raise ValueError(f"sometimes={sometimes}, type != {type(sometimes)}")
        res["sometimes"] = sometimes
    res["python"] = albumentations_to_python(res)
    return res


def albumentations_to_python(aug_info):
    pstr = ""
    for name, value in aug_info["params"].items():
        v = value
        if type(v) is list:  #name != 'nb_iterations' and
            v = (v[0], v[1])
        if type(value) is str:
            pstr += f"{name}='{v}', "
        else:
            pstr += f"{name}={v}, "

    if "sometimes" in aug_info and prob_already_add not in pstr:
        if pstr == "":
            res = f"A.{aug_info['name']}(p={aug_info['sometimes']})"
        else:
            res = f"A.{aug_info['name']}({pstr[:-2]}, p={aug_info['sometimes']})"
    else:
        res = f"A.{aug_info['name']}({pstr[:-2]})"

    return res


def get_function(aug_name):
    try:
        aug_f = getattr(A, aug_name)
        return aug_f
    except Exception as e:
        logger.error(repr(e))
        # raise e
        return None


def build_pipeline(aug_infos):
    global unsupport_bbox
    pipeline = []
    for aug_info in aug_infos:
        aug_name = aug_info["name"]
        params = aug_info["params"]

        aug_func = get_function(aug_name)
        sometimes = aug_info.get("sometimes", None)
        if sometimes is not None:
            params["p"] = sometimes
        aug = aug_func(**params)
        if type(aug) in unsupported_bbox_albs:
            unsupport_bbox = True

        pipeline.append(aug)

    if unsupport_bbox:
        augs = A.Compose(pipeline)
    else:
        augs = A.Compose(pipeline, bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    return augs


def build(aug_info):
    return build_pipeline([aug_info])


def _apply(transform, img, masks=None, coco_bboxes=None, class_labels=None):
    global unsupport_bbox

    if unsupport_bbox:
        transformed = transform(image=img, masks=masks)
        res_bboxes = []
        res_class_labels = []
    else:
        transformed = transform(image=img, masks=masks, bboxes=coco_bboxes, class_labels=class_labels)
        res_bboxes = transformed['bboxes']
        res_class_labels = transformed['class_labels']

    res_img = transformed['image']
    res_masks = transformed['masks']
    unsupport_bbox = False

    return res_img, res_masks, res_bboxes, res_class_labels


def apply(augs, meta: ProjectMeta, img, ann: Annotation):
    # @TODO: save object tags

    # works for rectangles
    det_meta, det_mapping = meta.to_detection_task(convert_classes=False)
    seg_meta, seg_mapping = meta.to_segmentation_task()
    res_meta = det_meta.merge(seg_meta)  # TagMetas should be preserved

    masks = []
    coco_bboxes = []
    class_labels = []
    rect_labels_names = {}
    index_to_class = {}
    bitmap_idx = 0
    for label in ann.labels:
        if type(label.geometry) in [Bitmap, Polygon]:
            image_mask = np.zeros((img.shape[0], img.shape[1]))
            label.geometry.draw(image_mask, color=1)
            masks.append(image_mask)
            index_to_class[bitmap_idx] = label.obj_class.name
            bitmap_idx += 1
        elif type(label.geometry) == Rectangle:
            curr_coco_bbox = get_coco_bbox(label.geometry)
            class_labels.append(label.obj_class.name)
            coco_bboxes.append(curr_coco_bbox)
            if label.obj_class.name not in rect_labels_names.keys():
                rect_labels_names[label.obj_class.name] = label

    res_img, res_masks, res_bboxes, res_class_labels = _apply(augs, img=img, masks=masks, coco_bboxes=coco_bboxes,
                                                              class_labels=class_labels)

    res_ann = Annotation.from_albumentations(index_to_class, res_img, res_bboxes, res_masks, res_class_labels,
                                             rect_labels_names, res_meta)

    return res_meta, res_img, res_ann


def pipeline_to_python(aug_infos):
    global unsupport_bbox

    if unsupport_bbox:
        template = \
            """import albumentations as A
    
            transform = A.Compose([
            {}
            ])
            """
    else:
        template = \
            """import albumentations as A
    
            transform = A.Compose([
            {}
            ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
            """
    py_lines = []
    for info in aug_infos:
        line = albumentations_to_python(info)
        _validate = info["python"]
        if line != _validate:
            raise ValueError("Generated python line differs from the one from config: \n\n{!r}\n\n{!r}"
                             .format(line, _validate))
        py_lines.append(line)
    res = template.format('\t' + ',\n\t'.join(py_lines))
    return res
