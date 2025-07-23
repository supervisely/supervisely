# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os.path as osp

import numpy as np
from scipy.io import loadmat

from fast_reid.fastreid.data.datasets import DATASET_REGISTRY

from .bases import Dataset


@DATASET_REGISTRY.register()
class PA100K(Dataset):
    """Pedestrian attribute dataset.
    80k training images + 20k test images.
    The folder structure should be:
        pa100k/
            data/ # images
            annotation.mat
    """
    dataset_dir = 'PA-100K'

    def __init__(self, root='', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.anno_mat_path = osp.join(
            self.dataset_dir, "annotation.mat"
        )

        required_files = [self.data_dir, self.anno_mat_path]
        self.check_before_run(required_files)

        train, val, test, attr_dict = self.extract_data()
        super(PA100K, self).__init__(train, val, test, attr_dict=attr_dict, **kwargs)

    def extract_data(self):
        # anno_mat is a dictionary with keys: ['test_images_name', 'val_images_name',
        # 'train_images_name', 'val_label', 'attributes', 'test_label', 'train_label']
        anno_mat = loadmat(self.anno_mat_path)

        def _extract(key_name, key_label):
            names = anno_mat[key_name]
            labels = anno_mat[key_label]
            num_imgs = names.shape[0]
            data = []
            for i in range(num_imgs):
                name = names[i, 0][0]
                attrs = labels[i, :].astype(np.float32)
                img_path = osp.join(self.data_dir, name)
                data.append((img_path, attrs))
            return data

        train = _extract('train_images_name', 'train_label')
        val = _extract('val_images_name', 'val_label')
        test = _extract('test_images_name', 'test_label')
        attrs = anno_mat['attributes']
        attr_dict = {i: str(attr[0][0]) for i, attr in enumerate(attrs)}

        return train, val, test, attr_dict
