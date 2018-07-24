# coding: utf-8

import os
import os.path as osp

import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib import logger


# returns mapping: (r, g, b) color -> some (row, col) for each unique color except black
def get_col2coord(img):
    img = img.astype(np.int32)
    h, w = img.shape[:2]
    colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
    unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
    indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
    col2coord = {(col // (256 ** 2), (col // 256) % 256, col % 256): (indx // w, indx % w)
                 for col, indx in col2indx.items()
                 if col != 0}
    return col2coord


class ImporterPascalVOCSegm:
    default_cls2col = {
        'neutral': (224, 224, 192),
        'aeroplane': (128, 0, 0),
        'bicycle': (0, 128, 0),
        'bird': (128, 128, 0),
        'boat': (0, 0, 128),
        'bottle': (128, 0, 128),
        'bus': (0, 128, 128),
        'car': (128, 128, 128),
        'cat': (64, 0, 0),
        'chair': (192, 0, 0),
        'cow': (64, 128, 0),
        'diningtable': (192, 128, 0),
        'dog': (64, 0, 128),
        'horse': (192, 0, 128),
        'motorbike': (64, 128, 128),
        'person': (192, 128, 128),
        'pottedplant': (0, 64, 0),
        'sheep': (128, 64, 0),
        'sofa': (0, 192, 0),
        'train': (128, 192, 0),
        'tvmonitor': (0, 64, 128),
    }

    def _read_datasets(self):
        self.src_datasets = {}
        for fname in os.listdir(self.lists_dir):
            if fname.endswith('.txt'):
                ds_name = osp.splitext(fname)[0]
                fpath = osp.join(self.lists_dir, fname)
                sample_names = list(filter(None, map(str.strip, open(fpath, 'r').readlines())))
                self.src_datasets[ds_name] = sample_names
                logger.info('Found source dataset "{}" with {} sample(s).'.format(ds_name, len(sample_names)))

    def _read_colors(self):
        if osp.isfile(self.colors_file):
            logger.info('Will try to read segmentation colors from provided file.')
            in_lines = filter(None, map(str.strip, open(self.colors_file, 'r').readlines()))
            in_splitted = (x.split() for x in in_lines)
            self.cls2col = {x[0]: (int(x[1]), int(x[2]), int(x[3])) for x in in_splitted}
            # fmt "NAME R G B" with values in [0; 255]
        else:
            logger.info('Will use default PascalVOC color mapping.')
            self.cls2col = self.default_cls2col
        logger.info('Determined {} class(es).'.format(len(self.cls2col)), extra={'classes': list(self.cls2col.keys())})
        self.col2cls = {v: k for k, v in self.cls2col.items()}

    @classmethod
    def _read_img(cls, img_path):
        img = cv2.imread(img_path).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def __init__(self):
        task_paths = sly.DtlPaths()
        self.in_dir = task_paths.data_dir
        self.out_dir = task_paths.results_dir
        self.settings = sly.json_load(task_paths.settings_path)

        self.lists_dir = osp.join(self.in_dir, 'ImageSets/Segmentation')
        self.imgs_dir = osp.join(self.in_dir, 'JPEGImages')
        self.segm_dir = osp.join(self.in_dir, 'SegmentationClass')
        self.inst_dir = osp.join(self.in_dir, 'SegmentationObject')
        self.colors_file = osp.join(self.in_dir, 'colors.txt')

        self.with_instances = osp.isdir(self.inst_dir)
        logger.info('Will import data {} instance info.'.format('with' if self.with_instances else 'without'))

        self._read_datasets()
        self._read_colors()

    def _get_ann(self, segm_path, inst_path, log_dct):
        segmentation_img = self._read_img(segm_path)

        if inst_path is not None:
            instance_img = self._read_img(inst_path)
            colored_img = instance_img

            instance_img16 = instance_img.astype(np.uint16)
            col2coord = get_col2coord(instance_img16)
            curr_col2cls = ((col, self.col2cls.get(tuple(segmentation_img[coord])))
                            for col, coord in col2coord.items())
            curr_col2cls = {k: v for k, v in curr_col2cls if v is not None}  # _instance_ color -> class name
        else:
            colored_img = segmentation_img
            curr_col2cls = self.col2cls  # class color -> class name

        imsize_wh = colored_img.shape[:2][::-1]
        figures = []
        for color, class_name in curr_col2cls.items():
            mask = np.all(colored_img == color, axis=2)  # exact match (3-channel img & rgb color)
            objs = sly.FigureBitmap.from_mask(class_name, imsize_wh, (0, 0), mask)
            figures.extend(objs)
            colored_img[mask] = (0, 0, 0)  # to check missing colors, see below

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
                    'src_img_path': osp.join(self.imgs_dir, name + '.jpg'),
                    'segm_path': osp.join(self.segm_dir, name + '.png'),
                }
                if self.with_instances:
                    dt['inst_path'] = osp.join(self.inst_dir, name + '.png')

                if all((osp.isfile(x) for x in dt.values())):
                    dt['image_ext'] = '.jpg'
                    out_pr.add_item(ds_name, name, dt)

        out_pr_fs = sly.ProjectFS(self.out_dir, out_pr)
        out_pr_fs.make_dirs()

        res_meta = sly.ProjectMeta()
        for color, class_name in self.col2cls.items():
            res_meta.classes.add({'title': class_name, 'shape': 'bitmap', 'color': sly.color2code(color)})
        res_meta.to_dir(out_pr_fs.project_path)

        progress = sly.progress_counter_import(out_pr.name, out_pr.image_cnt)
        for sample_info in out_pr_fs:
            self._convert_sample(sample_info)
            progress.iter_done_report()


def main():
    importer = ImporterPascalVOCSegm()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('PASCAL_VOC_IMPORT', main)
