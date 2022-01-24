# coding: utf-8
import os


def flag_from_env(s):
    return s.upper() in ['TRUE', 'YES', '1']


def remap_gpu_devices(in_device_ids):
    """
    Working limitation for CUDA
    :param in_device_ids: real GPU devices indexes. e.g.: [3, 4, 7]
    :return: CUDA ordered GPU indexes, e.g.: [0, 1, 2]
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, in_device_ids))
    return list(range(len(in_device_ids)))
