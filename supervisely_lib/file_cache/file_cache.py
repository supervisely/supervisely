# # coding: utf-8
# import os
# from supervisely_lib.file_cache.fs_caches import NNStorage, ImageStorage
#
#
# class FileCache:
#     def __init__(self, storage_root):
#         self._common_dir = constants.AGENT_STORAGE_DIR()
#
#         def create_st(stor_type, **kwargs):
#             if constants.WITH_LOCAL_STORAGE():
#                 sly.fs.mkdir(kwargs['storage_root'])
#                 return stor_type(**kwargs)
#             else:
#                 return EmptyStorage(**kwargs)
#
#         self._nn = create_st(NNStorage, name='NNs', storage_root=os.path.join(self._common_dir, 'neural_networks'))
#         self._image = create_st(ImageStorage, name='ImgCache', storage_root=os.path.join(self._common_dir, 'images'))
#
#     @property
#     def nn(self):
#         return self._nn
#
#     @property
#     def image(self):
#         return self._image