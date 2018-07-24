# coding: utf-8

import random

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
import supervisely_lib as sly


input_image_normalizer = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PytorchSlyDataset(Dataset):
    catcher_retries = 100

    def __init__(self, project_meta, samples, out_size_wh, class_mapping, bkg_color, allow_corrupted_cnt):
        self.project_meta = project_meta
        self.samples = samples
        self.out_size_wh = tuple(out_size_wh)
        self.class_mapping = class_mapping
        self.bkg_color = bkg_color
        self.sample_catcher = sly.CorruptedSampleCatcher(allow_corrupted_cnt)

    def __len__(self):
        return len(self.samples)

    def _get_sample_impl(self, img_fpath, ann_fpath):
        img = cv2.imread(img_fpath)[:, :, ::-1]  # raises exc if img is None
        h, w = img.shape[:2]
        img = cv2.resize(img, self.out_size_wh)

        ann_packed = sly.json_load(ann_fpath)
        ann = sly.Annotation.from_packed(ann_packed, self.project_meta)
        # ann.normalize_figures()  # @TODO: enaaaable!
        # will not resize figures: resize gt instead

        gt = np.ones((h, w), dtype=np.uint8) * self.bkg_color  # default bkg
        for fig in ann['objects']:
            gt_color = self.class_mapping.get(fig.class_title, None)
            if gt_color is None:
                raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(fig.class_title))
            fig.draw(gt, gt_color)

        gt = sly.resize_inter_nearest(gt, self.out_size_wh).astype(np.int64)
        gt = np.expand_dims(gt, 0)
        return input_image_normalizer(img), gt

    def __getitem__(self, idx):
        for att in range(self.catcher_retries):
            descr = self.samples[idx]
            res = self.sample_catcher.exec(idx, {'img': descr.img_path, 'ann': descr.ann_path},
                                           self._get_sample_impl, descr.img_path, descr.ann_path)
            if res is not None:
                return res
            idx = random.randrange(len(self.samples))  # must be ok for large ds
        raise RuntimeError('Unable to load some correct sample.')
