import inspect
import imgaug.augmenters as iaa
from collections import OrderedDict
from supervisely_lib.sly_logger import logger
from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.project.project_meta import ProjectMeta


#import supervisely_lib as sly
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np


def create_aug_info(category_name, aug_name, params, sometimes: float = None):
    clean_params = params
    #clean_params = remove_unexpected_arguments(category_name, aug_name, params)
    res = {
        "category": category_name,
        "name": aug_name,
        "params": clean_params,
    }
    if sometimes is not None:
        if type(sometimes) is not float or not (0.0 <= sometimes <= 1.0):
            raise ValueError(f"sometimes={sometimes}, type != {type(sometimes)}")
        res["sometimes"] = sometimes
    res["python"] = aug_to_python(res)
    return res


def aug_to_python(aug_info):
    pstr = ""
    for name, value in aug_info["params"].items():
        v = value
        if type(v) is list:  #name != 'nb_iterations' and
            v = (v[0], v[1])
        if type(value) is str:
            pstr += f"{name}='{v}', "
        else:
            pstr += f"{name}={v}, "
    method_py = f"iaa.{aug_info['category']}.{aug_info['name']}({pstr[:-2]})"

    res = method_py
    if "sometimes" in aug_info:
        res = f"iaa.Sometimes({aug_info['sometimes']}, {method_py})"
    return res


def pipeline_to_python(aug_infos, random_order=False):
    template = \
"""import imgaug.augmenters as iaa

seq = iaa.Sequential([
{}
], random_order={})
"""
    py_lines = []
    for info in aug_infos:
        line = aug_to_python(info)
        _validate = info["python"]
        if line != _validate:
            raise ValueError("Generated python line differs from the one from config: \n\n{!r}\n\n{!r}"
                             .format(line, _validate))
        py_lines.append(line)
    res = template.format('\t' + ',\n\t'.join(py_lines), random_order)
    return res


def get_default_params_by_function(f):
    params = []
    method_info = inspect.signature(f)
    for param in method_info.parameters.values():
        formatted = str(param)
        if 'deprecated' in formatted or 'seed=None' in formatted or 'name=None' in formatted:
            continue
        if param.default == inspect._empty:
            continue
        params.append({
            "pname": param.name,
            "default": param.default
        })
    return params


def get_default_params_by_name(category_name, aug_name):
    func = get_function(category_name, aug_name)
    defaults = get_default_params_by_function(func)
    return defaults


def get_function(category_name, aug_name):
    try:
        submodule = getattr(iaa, category_name)
        aug_f = getattr(submodule, aug_name)
        return aug_f
    except Exception as e:
        logger.error(repr(e))
        # raise e
        return None


def build_pipeline(aug_infos, random_order=False):
    pipeline = []
    for aug_info in aug_infos:
        category_name = aug_info["category"]
        aug_name = aug_info["name"]
        params = aug_info["params"]

        aug_func = get_function(category_name, aug_name)
        # TODO: hotfix:
        if aug_name == "CropAndPad":
            params["percent"] = tuple(params["percent"])

        aug = aug_func(**params)

        sometimes = aug_info.get("sometimes", None)
        if sometimes is not None:
            aug = iaa.meta.Sometimes(sometimes, aug)
        pipeline.append(aug)
    augs = iaa.Sequential(pipeline, random_order=random_order)
    return augs


def build(aug_info):
    return build_pipeline([aug_info])


def remove_unexpected_arguments(category_name, aug_name, params):
    # to avoid this:
    # TypeError: f() got an unexpected keyword argument 'b'
    defaults = get_default_params_by_name(category_name, aug_name)
    allowed_names = [d["pname"] for d in defaults]

    res = OrderedDict()
    for name, value in params.items():
        if name in allowed_names:
            res[name] = value
    return res


def _apply(augs: iaa.Sequential, img, boxes=None, masks=None):
    res = augs(images=[img], bounding_boxes=boxes, segmentation_maps=masks)
    #return image, boxes, masks
    return res[0][0], res[1], res[2]


def apply(augs, meta: ProjectMeta, img, ann: Annotation):
    # @TODO: save object tags

    # works for rectangles
    det_meta, det_mapping = meta.to_detection_task(convert_classes=False)
    det_ann = ann.to_detection_task(det_mapping)
    ia_boxes = det_ann.bboxes_to_imgaug()

    # works for polygons and bitmaps
    seg_meta, seg_mapping = meta.to_segmentation_task()
    seg_ann = ann.to_nonoverlapping_masks(seg_mapping)
    seg_ann = seg_ann.to_segmentation_task()
    class_to_index = {obj_class.name: idx for idx, obj_class in enumerate(seg_meta.obj_classes, start=1)}
    index_to_class = {v: k for k, v in class_to_index.items()}
    ia_masks = seg_ann.masks_to_imgaug(class_to_index)

    res_meta = det_meta.merge(seg_meta)  # TagMetas should be preserved

    res_img, res_ia_boxes, res_ia_masks = _apply(augs, img, ia_boxes, ia_masks)
    res_ann = Annotation.from_imgaug(res_img,
                                     ia_boxes=res_ia_boxes, ia_masks=res_ia_masks,
                                     index_to_class=index_to_class, meta=res_meta)
    # add image tags
    res_ann = res_ann.clone(img_tags=ann.img_tags)
    return res_meta, res_img, res_ann


def apply_to_image(augs, img):
    res_img, _, _ = _apply(augs, img, None, None)
    return res_img


def apply_to_image_and_mask(augs, img, mask):
    segmaps = SegmentationMapsOnImage(mask, shape=img.shape[:2])
    res_img, _, res_segmaps = _apply(augs, img, masks=segmaps)
    res_mask = res_segmaps.get_arr()
    if res_img.shape[:2] != res_mask.shape[:2]:
        raise ValueError(f"Image and mask have different shapes "
                         f"({res_img.shape[:2]} != {res_mask.shape[:2]}) after augmentations. "
                         f"Please, contact tech support")
    return res_img, res_mask

