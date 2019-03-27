# coding: utf-8


import os.path as osp

import supervisely_lib as sly

from worker import constants
from worker.fs_storages import ImageStorage, NNStorage, EmptyStorage


# may be created by different processes (no sync on agent side)
class AgentStorage:
    def __init__(self):
        self._common_dir = constants.AGENT_STORAGE_DIR()

        def create_st(stor_type, **kwargs):
            if constants.WITH_LOCAL_STORAGE():
                sly.fs.mkdir(kwargs['storage_root'])
                return stor_type(**kwargs)
            else:
                return EmptyStorage(**kwargs)

        self._nns = create_st(NNStorage,
                              name='NNs',
                              storage_root=osp.join(self._common_dir, 'models'))

        self._images = create_st(ImageStorage,
                                 name='ImgCache',
                                 storage_root=osp.join(self._common_dir, 'images'))

    @property
    def nns(self):
        return self._nns

    @property
    def images(self):
        return self._images
