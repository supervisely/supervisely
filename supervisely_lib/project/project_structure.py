# coding: utf-8

import os
import os.path as osp
from shutil import copyfile
from collections import namedtuple

from ..utils.os_utils import mkdir, ensure_base_path
from ..utils.json_utils import json_dump


class DatasetStructure:
    def __init__(self, name):
        # image_name -> ia_data dict
        # image_name means title (filename w/out ext), not DB id
        # ia_data dictionary stores any related data (image_ext, img_ann_id, img_data_id, ann etc)
        self.img_anns = {}
        self.name = name

    def __next__(self):
        yield from self.img_anns.items()  # image_name, ia_data

    def __iter__(self):
        return next(self)

    def __getitem__(self, item):
        return self.img_anns[item]

    @property
    def image_cnt(self):
        return len(self.img_anns)

    def get_ia_data(self, image_name, default=None):
        return self.img_anns.get(image_name, default)

    # does not add new img_anns
    def update_ia_data(self, rhs):
        for k, v in self.img_anns.items():
            rhs_ia_data = rhs.img_anns.get(k, None)
            if rhs_ia_data is not None:
                v.update(rhs_ia_data)


class ProjectStructure:
    IterItem = namedtuple('IterItem', [
        'project_name', 'ds_name', 'image_name', 'ia_data',
    ])

    def __init__(self, name):
        # ds_name -> DatasetStructure
        self.datasets = {}
        self.name = name

    def __next__(self):
        for ds_name, ds in self.datasets.items():
            for image_name, ia_data in ds:
                res = self.IterItem(project_name=self.name, ds_name=ds_name,
                                    image_name=image_name, ia_data=ia_data)
                yield res

    def __iter__(self):
        return next(self)

    def __getitem__(self, item):
        return self.datasets[item]

    @property
    def image_cnt(self):
        res = sum((x.image_cnt for x in self.datasets.values()))
        return res

    def get_ia_data(self, ds_name, image_name, default=None):
        ds = self.datasets.get(ds_name, None)
        if ds is None:
            return default
        return ds.get_ia_data(image_name, default)

    # does not add new img_anns or datasets
    def update_ia_data(self, rhs):
        for k, v in self.datasets.items():
            rhs_ds = rhs.datasets.get(k, None)
            if rhs_ds is not None:
                v.update_ia_data(rhs_ds)

    # silent replacement
    def add_item(self, ds_name, image_name, ia_data):
        if ds_name not in self.datasets:
            self.datasets[ds_name] = DatasetStructure(ds_name)  # don't use defaultdicts, they produce errs
        self.datasets[ds_name].img_anns[image_name] = ia_data


# ProjectFileStructure. Or ProjectFileSystem.
class ProjectFS:
    ann_extension = '.json'
    IterItem = namedtuple('IterItem', [
        'project_name', 'ds_name', 'image_name', 'ia_data', 'img_path', 'ann_path',
    ])

    def __init__(self, root_path, pr_structure):
        self.root_path = root_path
        self.pr_structure = pr_structure

    def __next__(self):
        for x in self.pr_structure:
            ann_fpath = self.ann_path(x.ds_name, x.image_name)
            img_fpath = self.img_path(x.ds_name, x.image_name)
            res = self.IterItem(project_name=x.project_name, ds_name=x.ds_name,
                                image_name=x.image_name, ia_data=x.ia_data,
                                img_path=img_fpath, ann_path=ann_fpath)
            yield res

    def __iter__(self):
        return next(self)

    @classmethod
    def _splitted_fnames_dct(cls, common_path):
        if not osp.isdir(common_path):
            return {}
        the_names = (f.name for f in os.scandir(common_path) if f.is_file())
        splitted_names = (osp.splitext(x) for x in the_names)
        res_dct = {k: v for k, v in splitted_names}
        return res_dct

    @property
    def image_cnt(self):
        return self.pr_structure.image_cnt

    @property
    def project_path(self):
        return osp.join(self.root_path, self.pr_structure.name)

    def dataset_path(self, ds_name):
        return osp.join(self.project_path, ds_name)

    def ann_fname(self, _, image_name):
        res = image_name + self.ann_extension
        return res

    def img_fname(self, ds_name, image_name):
        ia_data = self.pr_structure.get_ia_data(ds_name, image_name)
        img_ext = ia_data.get('image_ext')  # required field, with dot
        if img_ext is None:
            return None
        res = image_name + img_ext
        return res

    def ann_path(self, ds_name, image_name):
        ann_fname = self.ann_fname(ds_name, image_name)
        anns_path = self.dataset_anns_path(self.dataset_path(ds_name))
        return osp.join(anns_path, ann_fname)

    def img_path(self, ds_name, image_name):
        img_fname = self.img_fname(ds_name, image_name)
        if img_fname is None:
            return None
        imgs_path = self.dataset_imgs_path(self.dataset_path(ds_name))
        return osp.join(imgs_path, img_fname)

    # creates all required dirs for the structure
    def make_dirs(self):
        for ds_name in self.pr_structure.datasets.keys():
            ds_path = self.dataset_path(ds_name)
            mkdir(self.dataset_anns_path(ds_path))
            mkdir(self.dataset_imgs_path(ds_path))

    @classmethod
    def dataset_anns_path(cls, dataset_path):
        return osp.join(dataset_path, 'ann')

    @classmethod
    def dataset_imgs_path(cls, dataset_path):
        return osp.join(dataset_path, 'img')

    @classmethod
    def split_dir_project(cls, project_dir_path):
        root_path, project_name = osp.split(project_dir_path.rstrip('/'))
        if not project_name:
            raise RuntimeError('Unable to determine project name.')
        return root_path, project_name

    @classmethod
    def from_disk_dir_project(cls, project_dir_path):
        return cls.from_disk(*cls.split_dir_project(project_dir_path))

    # if by_annotations is set, some images may be omitted (image_ext will be None)
    @classmethod
    def from_disk(cls, root_path, project_name, by_annotations=False):
        fs = ProjectFS(root_path, ProjectStructure(project_name))
        possible_datasets = sorted([f.name for f in os.scandir(fs.project_path) if f.is_dir()])
        for ds_name in possible_datasets:
            dataset_path = fs.dataset_path(ds_name)
            anns_path = fs.dataset_anns_path(dataset_path)
            imgs_path = fs.dataset_imgs_path(dataset_path)
            if not osp.isdir(anns_path):
                continue  # skip dir, it is not a dataset
            if (not by_annotations) and (not osp.isdir(imgs_path)):
                continue  # skip dir, it is not a dataset

            new_ds = DatasetStructure(ds_name)
            ann_names = {k: v for k, v in cls._splitted_fnames_dct(anns_path).items()
                         if v == cls.ann_extension}  # by ext
            img_names = cls._splitted_fnames_dct(imgs_path)  # all files

            if by_annotations:
                name_set = set(ann_names.keys())
            else:
                name_set = set(ann_names.keys()) & set(img_names.keys())

            for image_name in name_set:
                image_ext = img_names.get(image_name)  # None for missing images
                new_ds.img_anns[image_name] = {'image_ext': image_ext}

            fs.pr_structure.datasets[ds_name] = new_ds

        return fs


class ProjectWriterFS:
    ann_extension = '.json'

    def __init__(self, init_dir, project_name=None):
        if project_name is None:
            self.pr_dir = init_dir   # init_dir is a project_dir
        else:
            self.pr_dir = osp.join(init_dir, project_name)  # init_dir is a parent_dir
            mkdir(self.pr_dir)

    def write_image(self, img_desc, free_name):
        img_path = self.get_img_path(img_desc.get_res_ds_name(), free_name, img_desc.get_image_ext())
        ensure_base_path(img_path)
        img_desc.write_image_local(img_path)

    def copy_image(self, img_desc, free_name):
        new_img_path = self.get_img_path(img_desc.get_res_ds_name(), free_name, img_desc.get_image_ext())
        ensure_base_path(new_img_path)
        orig_img_path = img_desc.get_img_path()
        copyfile(orig_img_path, new_img_path)

    def write_ann(self, img_desc, packed_ann, free_name):
        ann_path = self.get_ann_path(img_desc.get_res_ds_name(), free_name)
        ensure_base_path(ann_path)
        json_dump(packed_ann, ann_path)

    def write_meta(self, meta):
        meta.to_dir(self.pr_dir)

    def get_img_path(self, ds_name, file_name, file_ext):
        return os.path.join(self.pr_dir, ds_name, 'img', file_name + file_ext)

    def get_ann_path(self, ds_name, file_name):
        return os.path.join(self.pr_dir, ds_name, 'ann', file_name + ProjectWriterFS.ann_extension)


