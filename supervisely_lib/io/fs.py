# coding: utf-8

import os
import shutil
import errno
import tarfile
import subprocess
import requests
from requests.structures import CaseInsensitiveDict

from supervisely_lib._utils import get_bytes_hash, get_string_hash
from supervisely_lib.io.fs_cache import FileCache


def get_file_name(path: str) -> str:
    """
    Extracts file name from a given path.

    Args:
        path: File path.
    Returns:
         File name.
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_file_ext(path: str) -> str:
    """
    Extracts file extension from a given path.

    Args:
        path: File path.
    Returns:
         File extension.
    """
    return os.path.splitext(os.path.basename(path))[1]


def get_file_name_with_ext(path: str) -> str:
    """
    Extracts file name with ext from a given path.

    Args:
        path: File path.
    Returns:
         File name with ext.
    """
    return os.path.basename(path)


def list_dir_recursively(dir: str) -> list:
    """
    Recursively walks through directory and returns list with all file paths.

    Args:
        dir: Target dir path.
    Returns:
         A list containing file paths.
    """
    all_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_path = os.path.join(root, name)
            file_path = os.path.relpath(file_path, dir)
            all_files.append(file_path)
    return all_files


def list_files_recursively(dir: str, valid_extensions: list = None, filter_fn=None) -> list:
    """
    Recursively walks through directory and returns list with all file paths.

    Args:
        dir: Target dir path.
        valid_extensions:
        filter_fn: function with a single argument that determines whether to keep a given file path.
    Returns:
         A list containing file paths.
    """

    def file_path_generator():
        for dir_name, _, file_names in os.walk(dir):
            for filename in file_names:
                yield os.path.join(dir_name, filename)

    return [file_path for file_path in file_path_generator() if
            (valid_extensions is None or get_file_ext(file_path) in valid_extensions) and
            (filter_fn is None or filter_fn(file_path))]


def list_files(dir: str, valid_extensions: list = None, filter_fn=None) -> list:
    """
    Returns list with file paths presented in given directory.

    Args:
        dir: Target dir path.
        valid_extensions:
        filter_fn: function with a single argument that determines whether to keep a given file path.
    Returns:
         A list containing file paths.
    """
    res = list(os.path.join(dir, x.name) for x in os.scandir(dir) if x.is_file())
    return [file_path for file_path in res if
      (valid_extensions is None or get_file_ext(file_path) in valid_extensions) and
      (filter_fn is None or filter_fn(file_path))]


def mkdir(dir: str):
    """
    Creates a leaf directory and all intermediate ones.

    Args:
        dir: Target directory path.

    """
    os.makedirs(dir, mode=0o777, exist_ok=True)


# create base path recursively; pay attention to slash-terminating paths
def ensure_base_path(path):
    dst_dir = os.path.split(path)[0]
    if dst_dir:
        mkdir(dst_dir)


def copy_file(src: str, dst: str):
    """
    Args:
        src: Source file path.
        dst: Destination file path.

    """
    ensure_base_path(dst)
    with open(dst, 'wb') as out_f:
        with open(src, 'rb') as in_f:
            shutil.copyfileobj(in_f, out_f, length=1024 * 1024)


def hardlink_or_copy_file(src: str, dst: str):
    try:
        os.link(src, dst)
    except OSError:
        copy_file(src, dst)


def hardlink_or_copy_tree(src: str, dst: str):
    mkdir(dst)
    for dir_name, _, file_names in os.walk(src):
        relative_dir = os.path.relpath(dir_name, src)
        dst_sub_dir = os.path.join(dst, relative_dir)
        mkdir(dst_sub_dir)
        for file_name in file_names:
            hardlink_or_copy_file(os.path.join(dir_name, file_name), os.path.join(dst_sub_dir, file_name))


def dir_exists(dir: str) -> bool:
    """
    Check whether directory exists or not.

    Args:
        dir: Target directory path.
    Returns:
        True if directory exists, False otherwise.
    """
    return os.path.isdir(dir)


def dir_empty(dir: str) -> bool:
    """
    Check whether directory is empty or not.

    Args:
        dir: Target directory path.
    Returns:
        True if directory is empty, False otherwise.
    """
    if dir_exists(dir) and len(list_files_recursively(dir)) > 0:
        return False
    return True


def file_exists(path: str) -> bool:
    """
    Check whether file exists or not.

    Args:
        path: File path.
    Returns:
        True if file exists, False otherwise.
    """
    return os.path.isfile(path)


def get_subdirs(dir_path: str) -> list:
    """
    Return a list containing the names of the directories in the given directory.

    Args:
        dir_path: Target directory path.
    Returns:
        List containing directories names.
    """
    res = list(x.name for x in os.scandir(dir_path) if x.is_dir())
    return res


# removes directory content recursively
def clean_dir(dir_: str):
    """
    Recursively delete a directory tree, but save root directory.
    Args:
        dir_: Target directory path.
    """
    shutil.rmtree(dir_, ignore_errors=True)
    mkdir(dir_)


def remove_dir(dir_: str):
    """
    Recursively delete a directory tree.
    Args:
        dir_: Target directory path.
    """
    shutil.rmtree(dir_, ignore_errors=True)


def silent_remove(file_path: str):
    """
    Remove file which may not exist.
    Args:
        file_path: File path.
    """
    try:
        os.remove(file_path)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise


def get_file_size(path: str):
    """
    Return the size of a file, reported by os.stat().
    Args:
        path: File path.
    """
    return os.path.getsize(path)


def get_directory_size(dir_path: str) -> int:
    """
    Return the size of a directory.

    Args:
        dir_path: Target directory path.
    Returns:
         Directory size in bytes.
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += get_file_size(fp)
    return total_size


def archive_directory(dir_: str, tar_path: str):
    """
    Create tar archive from directory.
    Args:
        dir_: Target directory path.
        tar_path: Path for output tar archive.
    """
    with tarfile.open(tar_path, 'w', encoding='utf-8') as tar:
        tar.add(dir_, arcname=os.path.sep)


def get_file_hash(path):
    return get_bytes_hash(open(path, 'rb').read())


def tree(dir_path):
    out = subprocess.Popen(['tree', '--filelimit', '500', '-h', '-n', dir_path],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    return stdout.decode("utf-8")


def log_tree(dir_path, logger):
    out = tree(dir_path)
    logger.info("DIRECTORY_TREE", extra={'tree': out})


def touch(path):
    ensure_base_path(path)
    with open(path, 'a'):
        os.utime(path, None)


def download(url, save_path, cache: FileCache = None, progress=None):
    def _download():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(CaseInsensitiveDict(r.headers).get('Content-Length', '0'))
            if progress is not None:
                progress.set(0, total_size_in_bytes)
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    if progress is not None:
                        progress.iters_done_report(len(chunk))

    if cache is None:
        _download()
    else:
        cache_path = cache.check_storage_object(get_string_hash(url), get_file_ext(save_path))
        if cache_path is None:
            # file not in cache
            _download()
            cache.write_object(save_path, get_string_hash(url))
        else:
            cache.read_object(get_string_hash(url), save_path)
            if progress is not None:
                progress.set(0, get_file_size(save_path))
                progress.iters_done_report(get_file_size(save_path))

    return save_path