# coding: utf-8

import os
from os.path import join

import cv2
import numpy as np
from PIL import Image
import supervisely_lib as sly


class ImporterBinaryMasks:
    def __init__(self):
        task_paths = sly.DtlPaths()
        self.in_dir = task_paths.data_dir
        self.out_dir = task_paths.results_dir
        self.settings = sly.json_load(task_paths.settings_path)

        if not os.path.exists(join(self.in_dir, 'img')) or not os.path.exists((join(self.in_dir, 'ann'))):
            raise RuntimeError('Invalid data format. Input folder should contain "img" and "ann" dirs')
        self.img_dir = join(self.in_dir, 'img')
        self.ann_dir = join(self.in_dir, 'ann')

    def _get_ann(self, image_id, masks_map, image_size_wh):
        figures = []

        if image_id in masks_map:
            mask = cv2.imread(join(self.ann_dir, masks_map[image_id]))[:, :, 0]
            bool_mask = mask > 0
            figures.extend(sly.FigureBitmap.from_mask('untitled', image_size_wh, (0, 0), bool_mask))

        ann = sly.Annotation.new_with_objects(image_size_wh, figures)
        return ann

    def _convert_sample(self, sample_info, masks_map):
        image_name = sample_info.image_name
        image_ext_src = sample_info.ia_data['image_ext_in']
        src_image_path = join(self.img_dir, image_name + image_ext_src)

        src_image = np.array(Image.open(src_image_path))

        image_size_wh = src_image.shape[:2][::-1]
        cv2.imwrite(sample_info.img_path, src_image[:, :, ::-1])
        ann = self._get_ann(image_name, masks_map, image_size_wh)
        packed_ann = ann.pack()
        sly.json_dump(packed_ann, sample_info.ann_path)

    @staticmethod
    def _get_files_list(dir_path):
        images_list = os.listdir(dir_path)
        return images_list

    def convert(self):
        images_pathes = self._get_files_list(self.img_dir)
        masks_pathes = self._get_files_list(self.ann_dir)
        masks_map = {os.path.splitext(mask_p)[0]: mask_p for mask_p in masks_pathes}
        dataset_name = 'ds'
        out_pr = sly.ProjectStructure(self.settings['res_names']['project'])

        for image_fp in images_pathes:
            image_ext = os.path.splitext(image_fp)[1]
            image_id = os.path.splitext(image_fp)[0]
            dt = {
                "image_ext": ".png",
                "image_ext_in": image_ext
            }
            out_pr.add_item(dataset_name, image_id, dt)

        out_pr_fs = sly.ProjectFS(self.out_dir, out_pr)
        out_pr_fs.make_dirs()

        res_meta = sly.ProjectMeta()
        res_meta.classes.add({'title': 'untitled', 'shape': 'bitmap', 'color': sly.gen_new_color()})
        res_meta.to_dir(out_pr_fs.project_path)

        progress = sly.progress_counter_import(out_pr.name, out_pr.image_cnt)
        for sample_info in out_pr_fs:
            self._convert_sample(sample_info, masks_map)
            progress.iter_done_report()


def main():
    importer = ImporterBinaryMasks()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('BINARY_MASKS_IMPORT', main)
