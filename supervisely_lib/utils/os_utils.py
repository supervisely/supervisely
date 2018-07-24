# coding: utf-8

import os
import os.path as osp
import errno
import shutil
import hashlib
import base64
import tarfile


# remove file which may not exist
def silent_remove(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise


# make dirs for path recursively, w/out exception if dir already exists
def mkdir(dpath):
    os.makedirs(dpath, mode=0o777, exist_ok=True)


# create base path recursively; pay attention to slash-terminating paths
def ensure_base_path(path):
    dst_dir = osp.split(path)[0]
    mkdir(dst_dir)


def required_env(name):
    res = os.getenv(name, None)
    if res is None:
        raise RuntimeError('Wrong or missing env {}.'.format(name))
    return res


# working limitation for CUDA (pytorch, tf etc)
def remap_gpu_devices(in_device_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, in_device_ids))
    device_ids = list(range(len(in_device_ids)))
    return device_ids


def get_subdirs(dir_path):
    res = list(x.name for x in os.scandir(dir_path) if x.is_dir())
    return res


# removes directory content recursively
def clean_dir(dir_):
    shutil.rmtree(dir_, ignore_errors=True)
    mkdir(dir_)


def remove_dir(dir_):
    shutil.rmtree(dir_)


def get_image_hash(img_path):
    return base64.b64encode(hashlib.sha256(open(img_path, 'rb').read()).digest()).decode('utf-8')


def get_file_size(path):
    return os.path.getsize(path)


def get_file_ext(path):
    return os.path.splitext(path)[1]


def list_dir(dir_):
    all_files = []
    for root, dirs, files in os.walk(dir_):
        for name in files:
            file_path = os.path.join(root, name)
            file_path = os.path.relpath(file_path, dir_)
            all_files.append(file_path)
    return all_files


# without chunks, loads full file into memory
def copy_file(src, dst):
    with open(dst, 'wb') as out_f:
        with open(src, 'rb') as in_f:
            buff = in_f.read()
            out_f.write(buff)


def archive_directory(dir_, tar_path):
    with tarfile.open(tar_path, 'w', encoding='utf-8') as tar:
        tar.add(dir_, arcname=os.path.sep)


def file_exists(path):
    return os.path.isfile(path)
