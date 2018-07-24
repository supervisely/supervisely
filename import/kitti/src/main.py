# coding: utf-8

import os
import os.path as osp

import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger


# returns mapping: x (unit16) color -> some (row, col) for each unique color except black
def get_col2coord(img):
    h, w = img.shape[:2]
    unq, unq_inv, unq_cnt = np.unique(img, return_inverse=True, return_counts=True)
    indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
    col2coord = {col: (indx // w, indx % w) for col, indx in col2indx.items() if col != 0}
    return col2coord


class ImporterKittiSemSeg:
    def _read_datasets(self):
        self.src_datasets = {}
        ds_names = [x for x in os.listdir(self.in_dir) if osp.isdir(osp.join(self.in_dir, x))]
        for ds_name in ds_names:
            imgdir = self._imgs_dir(ds_name)
            sample_names = [osp.splitext(x)[0] for x in os.listdir(imgdir) if osp.isfile(osp.join(imgdir, x))]
            self.src_datasets[ds_name] = sample_names
            logger.info('Found source dataset "{}" with {} sample(s).'.format(ds_name, len(sample_names)))

    def _read_colors(self):
        if osp.isfile(self.colors_file):
            logger.info('Will try to read segmentation colors from provided file.')
            labels = sly.json_load(self.colors_file)
        else:
            logger.info('Will use default Kitti (Cityscapes) color mapping.')
            default_fpath = osp.join(osp.dirname(__file__), 'colors.json')
            labels = sly.json_load(default_fpath)
        self.instance_classes = [el['name'] for el in labels if el['hasInstances']]
        self.cls2col = {el['name']: tuple(el['color']) for el in labels}
        self.idx2cls = {el['id']: el['name'] for el in labels}
        logger.info('Determined {} class(es).'.format(len(self.cls2col)), extra={'classes': list(self.cls2col.keys())})
        self.cls_names = [el['name'] for el in labels]  # ! from order of labels
        # self.col2cls = {v: k for k, v in self.cls2col.items()}

    @classmethod
    def _read_img(cls, img_path):
        img = cv2.imread(img_path).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @classmethod
    def _read_img_unch(cls, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # expect uint16
        return img

    def _imgs_dir(self, ds_name):
        return osp.join(self.in_dir, ds_name, 'image_2')

    def _segm_dir(self, ds_name):
        return osp.join(self.in_dir, ds_name, 'semantic')

    def _inst_dir(self, ds_name):
        return osp.join(self.in_dir, ds_name, 'instance')

    def __init__(self):
        task_paths = sly.DtlPaths()
        self.in_dir = task_paths.data_dir
        self.out_dir = task_paths.results_dir
        self.settings = sly.json_load(task_paths.settings_path)
        self.colors_file = osp.join(self.in_dir, 'colors.json')
        self._read_colors()
        self._read_datasets()

    def _get_ann(self, segm_path, inst_path, log_dct):
        # segmentation_img = self._read_img(segm_path)  # unused

        instance_img = self._read_img_unch(inst_path)
        colored_img = instance_img

        col2coord = get_col2coord(instance_img)
        curr_col2cls = {col: self.idx2cls[int(col // 256)]  # some hack to determine class correctly
                        for col, coord in col2coord.items()}

        imsize_wh = colored_img.shape[:2][::-1]
        figures = []
        for color, class_name in curr_col2cls.items():
            mask = colored_img == color  # exact match for 1d uint16
            objs = sly.FigureBitmap.from_mask(class_name, imsize_wh, (0, 0), mask)
            figures.extend(objs)
            colored_img[mask] = 0  # to check missing colors, see below

        if np.sum(colored_img) > 0:
            logger.warn('Not all objects or classes are captured from source segmentation.', extra=log_dct)

        ann = sly.Annotation.new_with_objects(imsize_wh, figures)
        return ann

    def _convert_sample(self, sample_info):
        log_dct = sample_info._asdict()  # ok, it's documented
        # logger.trace('Will process sample.', extra=log_dct)

        sample_data = sample_info.ia_data
        sly.copy_file(sample_data['src_img_path'], sample_info.img_path)  # img is ready

        ann = self._get_ann(sample_data['segm_path'], sample_data.get('inst_path'), log_dct)
        packed_ann = ann.pack()
        sly.json_dump(packed_ann, sample_info.ann_path)  # ann is ready

    def convert(self):
        # map input structure to output
        out_pr = sly.ProjectStructure(self.settings['res_names']['project'])

        for ds_name, sample_names in self.src_datasets.items():
            for name in sample_names:
                dt = {
                    'src_img_path': osp.join(self._imgs_dir(ds_name), name + '.png'),
                    'segm_path': osp.join(self._segm_dir(ds_name), name + '.png'),
                    'inst_path':  osp.join(self._inst_dir(ds_name), name + '.png')
                }
                if all((osp.isfile(x) for x in dt.values())):
                    dt['image_ext'] = '.png'
                    out_pr.add_item(ds_name, name, dt)

        out_pr_fs = sly.ProjectFS(self.out_dir, out_pr)
        out_pr_fs.make_dirs()

        res_meta = sly.ProjectMeta()
        for class_name, color in self.cls2col.items():
            res_meta.classes.add({'title': class_name, 'shape': 'bitmap', 'color': sly.color2code(color)})
        res_meta.to_dir(out_pr_fs.project_path)

        progress = sly.progress_counter_import(out_pr.name, out_pr.image_cnt)
        for sample_info in out_pr_fs:
            self._convert_sample(sample_info)
            progress.iter_done_report()


def main():
    importer = ImporterKittiSemSeg()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('KITTI_SEM_SEG_IMPORT', main)
