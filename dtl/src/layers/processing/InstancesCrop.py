# coding: utf-8

from copy import deepcopy

from supervisely_lib import Rect, rect_from_bounds
from supervisely_lib.figure.aux import crop_image_with_rect

from Layer import Layer
from classes_utils import ClassConstants


class InstancesCropLayer(Layer):

    action = 'instances_crop'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["classes", "pad"],
                "properties": {
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "save_classes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "pad": {
                        "type": "object",
                        "required": ["sides"],
                        "properties": {
                            "sides": {
                                "type": "object",
                                "uniqueItems": True,
                                "items": {
                                    "type": "string",
                                    "patternProperties": {
                                        "(left)|(top)|(bottom)|(right)": {
                                            "type": "string",
                                            "pattern": "^[0-9]+(%)|(px)$"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)
        self.classes_to_crop, self.classes_to_save = self._get_cls_lists()
        if len(self.classes_to_crop) == 0:
            raise ValueError("InstancesCropLayer: classes array can not be empty")
        if len(set(self.classes_to_crop) & set(self.classes_to_save)) > 0:
            raise ValueError("InstancesCropLayer: classes and save_classes must not intersect")

    def _get_cls_lists(self):
        return self.settings['classes'], self.settings.get('save_classes', [])

    def requires_image(self):
        return True

    def define_classes_mapping(self):
        classes_to_crop, classes_to_save = self._get_cls_lists()
        for cls in classes_to_save + classes_to_crop:
            self.cls_mapping[cls] = ClassConstants.DEFAULT
        self.cls_mapping[ClassConstants.OTHER] = ClassConstants.IGNORE

    def process(self, data_el):
        img_desc, ann_orig = data_el
        imsize_wh = ann_orig.image_size_wh
        rect_img = Rect.from_size(imsize_wh)
        padding_dct = self.settings['pad']['sides']
        img_orig = None

        for idx, src_fig in enumerate(ann_orig['objects']):
            if src_fig.class_title not in self.classes_to_crop:
                continue

            src_fig_bbox = src_fig.get_bbox().round()
            if src_fig_bbox.is_empty:
                continue  # tiny object
            new_img_rect = rect_from_bounds(
                padding_dct, img_w=src_fig_bbox.width, img_h=src_fig_bbox.height, shift_inside=False
            )
            rect_to_crop = new_img_rect.move(src_fig_bbox.point0)
            rect_to_crop = rect_to_crop.intersection(rect_img).round()
            if rect_to_crop.is_empty:
                continue

            # let's crop
            if img_orig is None:
                img_orig = img_desc.read_image()

            img = crop_image_with_rect(img_orig, rect_to_crop)
            new_img_desc = img_desc.clone_with_img(img)

            ann = deepcopy(ann_orig)
            ann['objects'] = [x for i, x in enumerate(ann['objects'])
                              if i == idx or x.class_title in self.classes_to_save]
            ann.apply_to_figures(lambda x: x.crop(rect_to_crop))

            delta = (-rect_to_crop.left, -rect_to_crop.top)
            for fig in ann['objects']:
                fig.shift(delta)  # to new coords of image
            ann.update_image_size(img)

            yield new_img_desc, ann
