# coding: utf-8

import os.path as osp
from copy import deepcopy

import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib import ProjectWriterFS, FigureBitmap, FigurePolygon

from Layer import Layer


# save to archive
class SaveLayer(Layer):

    action = 'save'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["images", "annotations"],
                "properties": {
                    "images": {
                        "type": "boolean"
                    },
                    "annotations": {
                        "type": "boolean"
                    },
                    "visualize": {
                        "type": "boolean"
                    }
                }
            }
        }
    }

    @classmethod
    def draw_colored_mask(cls, ann, color_mapping):
        w, h = ann.image_size_wh
        line_w = int((max(w, h) + 1) / 300)
        line_w = max(line_w, 1)
        res_img = np.zeros((h, w, 3), dtype=np.uint8)

        for fig in ann['objects']:
            color = color_mapping.get(fig.class_title)
            if color is None:
                continue  # ignore now
            if isinstance(fig, FigureBitmap) or isinstance(fig, FigurePolygon):
                fig.draw(res_img, color)
            else:
                fig.draw_contour(res_img, color, line_w)

        return res_img

    def __init__(self, config, output_folder, net):
        Layer.__init__(self, config)
        if not self.settings['images'] and not self.settings['annotations']:
            raise ValueError("images or annotations should be set to true")

        self.output_folder = output_folder
        self.net = net
        self.pr_writer = ProjectWriterFS(output_folder)

    def is_archive(self):
        return True

    def requires_image(self):
        return True

    def process(self, data_el):
        img_desc, ann = data_el
        free_name = self.net.get_free_name(img_desc.get_img_name())
        new_dataset_name = img_desc.get_res_ds_name()

        if self.settings.get('visualize'):
            out_meta = self.net.get_result_project_meta()
            cls_mapping = {}
            for cls_descr in out_meta.classes:
                color_s = cls_descr.get('color')
                if color_s is not None:
                    color = sly.hex2rgb(color_s)
                else:
                    color = sly.get_random_color()
                cls_mapping[cls_descr['title']] = color

            # hack to draw 'black' regions
            cls_mapping = {k: (1, 1, 1) if max(v) == 0 else v for k, v in cls_mapping.items()}

            vis_img = self.draw_colored_mask(ann, cls_mapping)
            orig_img = img_desc.read_image()
            comb_img = sly.overlay_images(orig_img, vis_img, 0.5)

            sep = np.array([[[0, 255, 0]]] * orig_img.shape[0], dtype=np.uint8)
            img = np.hstack((orig_img, sep, comb_img))

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output_img_path = osp.join(self.output_folder, new_dataset_name, 'visualize', free_name + '.png')
            sly.ensure_base_path(output_img_path)
            cv2.imwrite(output_img_path, img)

        # net _always_ downloads images
        if self.settings['images'] is True:
            if img_desc.need_write() is True:
                self.pr_writer.write_image(img_desc, free_name)
            else:
                self.pr_writer.copy_image(img_desc, free_name)

        if self.settings['annotations'] is True:
            ann_to_save = deepcopy(ann)
            ann_to_save.normalize_figures()
            packed_ann = ann_to_save.pack()
            self.pr_writer.write_ann(img_desc, packed_ann, free_name)

        yield ([img_desc, ann],)
