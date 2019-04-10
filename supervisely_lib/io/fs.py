# coding: utf-8

import os
import shutil
import errno
import tarfile

from supervisely_lib._utils import get_bytes_hash


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


def list_files(dir: str, valid_extensions: list=None, filter_fn=None) -> list:
    """
    Recursively walks through directory and returns list with all file paths.

    Args:
        dir: Target dir path.
        valid_extensions:
        filter_fn: function with a single argument that determines whether to keep a given file path.
    Returns:
         A list containing file paths.
    """
    return [f.path for f in os.scandir(dir) if f.is_file() and
            (valid_extensions is None or get_file_ext(f.path) in valid_extensions) and
            (filter_fn is None or filter_fn(f.path))]


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
    mkdir(dst_dir)


def copy_file(src: str, dst: str):
    """
    Copy file without chunks, loads full file into memory.

    Args:
        src: Source file path.
        dst: Destination file path.

    """
    ensure_base_path(dst)
    with open(dst, 'wb') as out_f:
        with open(src, 'rb') as in_f:
            buff = in_f.read()
            out_f.write(buff)


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
    if dir_exists(dir) and len(list_files(dir)) > 0:
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