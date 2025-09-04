# coding: utf-8

# docs
import base64
import errno
import hashlib
import mimetypes
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import aiofiles
import requests
from requests.structures import CaseInsensitiveDict
from tqdm import tqdm

from supervisely._utils import get_bytes_hash, get_or_create_event_loop, get_string_hash

if TYPE_CHECKING:
    from supervisely.api.image_api import BlobImageInfo

from supervisely.io.fs_cache import FileCache
from supervisely.sly_logger import logger
from supervisely.task.progress import Progress

JUNK_FILES = [".DS_Store", "__MACOSX", "._.DS_Store", "Thumbs.db", "desktop.ini"]
OFFSETS_PKL_SUFFIX = "_offsets.pkl"  # suffix for pickle file with image offsets
OFFSETS_PKL_BATCH_SIZE = 10000  # 10k images per batch when loading from pickle


def get_file_name(path: str) -> str:
    """
    Extracts file name from a given path.

    :param path: Path to file.
    :type path: str
    :returns: File name without extension
    :rtype: :class:`str`
    :Usage example:

     .. code-block::

        import supervisely as sly

        file_name = sly.fs.get_file_name("/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg")

        print(file_name)
        # Output: IMG_0748
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_file_ext(path: str) -> str:
    """
    Extracts file extension from a given path.

    :param path: Path to file.
    :type path: str
    :returns: File extension without name
    :rtype: :class:`str`
    :Usage example:

     .. code-block::

        import supervisely as sly

        file_ext = sly.fs.get_file_ext("/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg")

        print(file_ext)
        # Output: .jpeg
    """
    return os.path.splitext(os.path.basename(path))[1]


def get_file_name_with_ext(path: str) -> str:
    """
    Extracts file name with ext from a given path.

    :param path: Path to file.
    :type path: str
    :returns: File name with extension
    :rtype: :class:`str`
    :Usage example:

     .. code-block::

        import supervisely as sly

        file_name_ext = sly.fs.get_file_name_with_ext("/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg")

        print(file_name_ext)
        # Output: IMG_0748.jpeg
    """
    return os.path.basename(path)


def remove_junk_from_dir(dir: str) -> List[str]:
    """
    Cleans the given directory from junk files and dirs (e.g. .DS_Store, __MACOSX, Thumbs.db, etc.).

    :param dir: Path to directory.
    :type dir: str
    :returns: List of global paths to removed files and dirs.
    :rtype: List[str]

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        input_dir = "/home/admin/work/projects/lemons_annotated/"
        sly.fs.remove_junk_from_dir(input_dir)
    """
    paths = list_dir_recursively(dir, include_subdirs=True, use_global_paths=True)
    removed_paths = []
    for path in paths:
        if get_file_name(path) in JUNK_FILES:
            if os.path.isfile(path):
                silent_remove(path)
                removed_paths.append(path)
            elif os.path.isdir(path):
                remove_dir(path)
                removed_paths.append(path)


def list_dir_recursively(
    dir: str, include_subdirs: bool = False, use_global_paths: bool = False
) -> List[str]:
    """
    Recursively walks through directory and returns list with all file paths, and optionally subdirectory paths.

    :param path: Path to directory.
    :type path: str
    :param include_subdirs: If True, subdirectory paths will be included in the result list.
    :type include_subdirs: bool
    :param use_global_paths: If True, absolute paths will be returned instead of relative ones.
    :type use_global_paths: bool
    :returns: List containing file paths, and optionally subdirectory paths.
    :rtype: :class:`List[str]`
    :Usage example:

     .. code-block::

        import supervisely as sly

        list_dir = sly.fs.list_dir_recursively("/home/admin/work/projects/lemons_annotated/")

        print(list_dir)
        # Output: ['meta.json', 'ds1/ann/IMG_0748.jpeg.json', 'ds1/ann/IMG_4451.jpeg.json', 'ds1/img/IMG_0748.jpeg', 'ds1/img/IMG_4451.jpeg']
    """
    all_paths = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_path = os.path.join(root, name)
            file_path = (
                os.path.relpath(file_path, dir)
                if not use_global_paths
                else os.path.abspath(file_path)
            )
            all_paths.append(file_path)
        if include_subdirs:
            for name in dirs:
                subdir_path = os.path.join(root, name)
                subdir_path = (
                    os.path.relpath(subdir_path, dir)
                    if not use_global_paths
                    else os.path.abspath(subdir_path)
                )
                all_paths.append(subdir_path)
    return all_paths


def list_files_recursively(
    dir: str,
    valid_extensions: Optional[List[str]] = None,
    filter_fn=None,
    ignore_valid_extensions_case: Optional[bool] = False,
) -> List[str]:
    """
    Recursively walks through directory and returns list with all file paths.
    Can be filtered by valid extensions and filter function.

    :param dir: Target dir path.
    :param dir: str
    :param valid_extensions: List with valid file extensions.
    :type valid_extensions: List[str], optional
    :param filter_fn: Function with a single argument. Argument is a file path. Function determines whether to keep a given file path. Must return True or False.
    :type filter_fn: Callable, optional
    :param ignore_valid_extensions_case: If True, validation of file extensions will be case insensitive.
    :type ignore_valid_extensions_case: bool
    :returns: List with file paths
    :rtype: :class:`List[str]`
    :Usage example:

     .. code-block:: python

         import supervisely as sly

         list_files = sly.fs.list_files_recursively("/home/admin/work/projects/lemons_annotated/ds1/img/")

         print(list_files)
         # Output: ['/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg', '/home/admin/work/projects/lemons_annotated/ds1/img/IMG_4451.jpeg']
    """

    def file_path_generator():
        for dir_name, _, file_names in os.walk(dir):
            for filename in file_names:
                yield os.path.join(dir_name, filename)

    valid_extensions = (
        valid_extensions
        if ignore_valid_extensions_case is False
        else [ext.lower() for ext in valid_extensions]
    )
    files = []
    for file_path in file_path_generator():
        file_ext = get_file_ext(file_path)
        if ignore_valid_extensions_case:
            file_ext.lower()
        if (valid_extensions is None or file_ext in valid_extensions) and (
            filter_fn is None or filter_fn(file_path)
        ):
            files.append(file_path)
    return files


def list_files(
    dir: str,
    valid_extensions: Optional[List[str]] = None,
    filter_fn=None,
    ignore_valid_extensions_case: Optional[bool] = False,
) -> List[str]:
    """
    Returns list with file paths presented in given directory.
    Can be filtered by valid extensions and filter function.
    Also can be case insensitive for valid extensions.

    :param dir: Target dir path.
    :param dir: str
    :param valid_extensions: List with valid file extensions.
    :type valid_extensions: List[str]
    :param filter_fn: Function with a single argument. Argument is a file path. Function determines whether to keep a given file path. Must return True or False.
    :type filter_fn: Callable, optional
    :param ignore_valid_extensions_case: If True, validation of file extensions will be case insensitive.
    :type ignore_valid_extensions_case: bool
    :returns: List with file paths
    :rtype: :class:`List[str]`
    :Usage example:

     .. code-block:: python

         import supervisely as sly

         list_files = sly.fs.list_files("/home/admin/work/projects/lemons_annotated/ds1/img/")

         print(list_files)
         # Output: ['/home/admin/work/projects/lemons_annotated/ds1/img/IMG_0748.jpeg', '/home/admin/work/projects/lemons_annotated/ds1/img/IMG_4451.jpeg']
    """
    res = list(os.path.join(dir, x.name) for x in os.scandir(dir) if x.is_file())

    files = []
    for file_path in res:
        file_ext = get_file_ext(file_path)

        if ignore_valid_extensions_case:
            file_ext = file_ext.lower()
            valid_extensions = [ext.lower() for ext in valid_extensions]

        if (valid_extensions is None or file_ext in valid_extensions) and (
            filter_fn is None or filter_fn(file_path)
        ):
            files.append(file_path)

    return files

    # return [
    #     file_path
    #     for file_path in res
    #     if (valid_extensions is None or get_file_ext(file_path) in valid_extensions)
    #     and (filter_fn is None or filter_fn(file_path))
    # ]


def mkdir(dir: str, remove_content_if_exists: Optional[bool] = False) -> None:
    """
    Creates a leaf directory and all intermediate ones.

    :param dir: Target dir path.
    :param dir: str
    :remove_content_if_exists: Remove directory content if it exist.
    :remove_content_if_exists: bool
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import mkdir
        mkdir('/home/admin/work/projects/example')
    """
    if dir_exists(dir) and remove_content_if_exists is True:
        clean_dir(dir, ignore_errors=True)
    else:
        os.makedirs(dir, mode=0o777, exist_ok=True)


# create base path recursively; pay attention to slash-terminating paths
def ensure_base_path(path: str) -> None:
    """
    Recursively create parent directory for target path.

    :param path: Target dir path.
    :type path: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import ensure_base_path
        ensure_base_path('/home/admin/work/projects/example')
    """
    dst_dir = os.path.split(path)[0]
    if dst_dir:
        mkdir(dst_dir)


def copy_file(src: str, dst: str) -> None:
    """
    Copy file from one path to another, if destination directory doesn't exist it will be created.

    :param src: Source file path.
    :type src: str
    :param dst: Destination file path.
    :type dst: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import copy_file
        copy_file('/home/admin/work/projects/example/1.png', '/home/admin/work/tests/2.png')
    """
    ensure_base_path(dst)
    with open(dst, "wb") as out_f:
        with open(src, "rb") as in_f:
            shutil.copyfileobj(in_f, out_f, length=1024 * 1024)


def hardlink_or_copy_file(src: str, dst: str) -> None:
    """
    Creates a hard link pointing to src named dst. If the link cannot be created, the file will be copied.

    :param src: Source file path.
    :type src: str
    :param dst: Destination file path.
    :type dst: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import hardlink_or_copy_file
        hardlink_or_copy_file('/home/admin/work/projects/example/1.png', '/home/admin/work/tests/link.txt')
    """
    try:
        os.link(src, dst)
    except OSError:
        copy_file(src, dst)


def hardlink_or_copy_tree(src: str, dst: str) -> None:
    """
    Creates a hard links pointing to src named dst files recursively. If the link cannot be created, the file will be copied.

    :param src: Source dir path.
    :type src: str
    :param dst: Destination dir path.
    :type dst: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import hardlink_or_copy_tree
        hardlink_or_copy_tree('/home/admin/work/projects/examples', '/home/admin/work/tests/links')
    """
    mkdir(dst)
    for dir_name, _, file_names in os.walk(src):
        relative_dir = os.path.relpath(dir_name, src)
        dst_sub_dir = os.path.join(dst, relative_dir)
        mkdir(dst_sub_dir)
        for file_name in file_names:
            hardlink_or_copy_file(
                os.path.join(dir_name, file_name), os.path.join(dst_sub_dir, file_name)
            )


def dir_exists(dir: str) -> bool:
    """
    Check whether directory exists or not.

    :param dir: Target directory path.
    :type dir: str
    :returns: True if directory exists, False otherwise.
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

          from supervisely.io.fs import dir_exists
          dir_exists('/home/admin/work/projects/examples') # True
          dir_exists('/home/admin/work/not_exist_dir') # False
    """
    return os.path.isdir(dir)


def dir_empty(dir: str) -> bool:
    """
    Check whether directory is empty or not.

    :param dir: Target directory path.
    :type dir: str
    :returns: True if directory is empty, False otherwise.
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

          from supervisely.io.fs import dir_empty
          dir_empty('/home/admin/work/projects/examples') # False
    """
    if dir_exists(dir) and len(list_files_recursively(dir)) > 0:
        return False
    return True


def file_exists(path: str) -> bool:
    """
    Check whether file exists or not.

    :param dir: Target file path.
    :type dir: str
    :returns: True if file exists, False otherwise.
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

          from supervisely.io.fs import file_exists
          file_exists('/home/admin/work/projects/examples/1.jpeg') # True
          file_exists('/home/admin/work/projects/examples/not_exist_file.jpeg') # False
    """
    return os.path.isfile(path)


def get_subdirs(dir_path: str, recursive: Optional[bool] = False) -> list:
    """
    Get list containing the names of the directories in the given directory.

    :param dir_path: Target directory path.
    :type dir_path: str
    :param recursive: If True, all found subdirectories will be included in the result list.
    :type recursive: bool
    :returns: List containing directories names.
    :rtype: :class:`list`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_subdirs
        subdirs = get_subdirs('/home/admin/work/projects/examples')
        print(subdirs)
        # Output: ['tests', 'users', 'ds1']
    """
    if recursive:
        return [
            global_to_relative(entry, dir_path)
            for entry in list_dir_recursively(dir_path, include_subdirs=True, use_global_paths=True)
            if os.path.isdir(entry)
        ]
    res = list(x.name for x in os.scandir(dir_path) if x.is_dir())
    return res


def get_subdirs_tree(dir_path: str) -> Dict[str, Union[str, Dict]]:
    """Returns a dictionary representing the directory tree.
    It will have only directories and subdirectories (not files).

    :param dir_path: Target directory path.
    :type dir_path: str
    :returns: Dictionary representing the directory tree.
    :rtype: :class:`Dict[str, Union[str, Dict]]`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_subdirs_tree
        tree = get_subdirs_tree('/home/admin/work/projects/examples')
        print(tree)
        # Output: {'examples': {'tests': {}, 'users': {}, 'ds1': {}}}
    """

    tree = {}
    subdirs = get_subdirs(dir_path, recursive=True)
    for subdir in subdirs:
        parts = subdir.split(os.sep)
        d = tree
        for part in parts:
            if part not in d:
                d[part] = {}
            d = d[part]

    return tree


def subdirs_tree(
    dir_path: str,
    ignore: Optional[List[str]] = None,
    ignore_content: Optional[List[str]] = None,
) -> Generator[str, None, None]:
    """Generator that yields directories in the directory tree,
    starting from the level below the root directory and then going down the tree.
    If ignore is specified, it will ignore paths which end with the specified directory names.
    All subdirectories of ignored directories will still be yielded.

    :param dir_path: Target directory path.
    :type dir_path: str
    :param ignore: List of directories to ignore. Note, that function still will yield
        subdirectories of ignored directories. It will only ignore paths which end with
        the specified directory names.
    :type ignore: List[str]
    :param ignore_content: List of directories which subdirectories should be ignored.
    :type ignore_content: List[str]
    :returns: Generator that yields directories in the directory tree.
    :rtype: Generator[str, None, None]
    """
    tree = get_subdirs_tree(dir_path)
    ignore = ignore or []
    ignore_content = ignore_content or []

    def _subdirs_tree(tree, path=""):
        for key, value in tree.items():
            new_path = os.path.join(path, key) if path else key
            if not any(new_path.endswith(i) for i in ignore):
                yield new_path
            if any(new_path.endswith(i) for i in ignore_content):
                continue
            if value:
                yield from _subdirs_tree(value, new_path)

    yield from _subdirs_tree(tree)


def global_to_relative(global_path: str, base_dir: str) -> str:
    """
    Converts global path to relative path.

    :param global_path: Global path.
    :type global_path: str
    :param base_dir: Base directory path.
    :type base_dir: str
    :returns: Relative path.
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import global_to_relative
        relative_path = global_to_relative('/home/admin/work/projects/examples/1.jpeg', '/home/admin/work/projects')
        print(relative_path)
        # Output: examples/1.jpeg
    """
    return os.path.relpath(global_path, base_dir)


# removes directory content recursively
def clean_dir(dir_: str, ignore_errors: Optional[bool] = True) -> None:
    """
    Recursively delete a directory tree, but save root directory.

    :param dir_: Target directory path.
    :type dir_: str
    :ignore_errors: Ignore possible errors while removes directory content.
    :ignore_errors: bool
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import clean_dir
        clean_dir('/home/admin/work/projects/examples')
    """
    # old implementation
    # shutil.rmtree(dir_, ignore_errors=True)
    # mkdir(dir_)

    for filename in os.listdir(dir_):
        file_path = os.path.join(dir_, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warn(f"Failed to delete {file_path}. Reason: {repr(e)}")
            if ignore_errors is False:
                raise e


def remove_dir(dir_: str) -> None:
    """
    Recursively delete a directory tree.

    :param dir_: Target directory path.
    :type dir_: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import remove_dir
        remove_dir('/home/admin/work/projects/examples')
    """
    shutil.rmtree(dir_, ignore_errors=True)


def silent_remove(file_path: str) -> None:
    """
    Remove file which may not exist.

    :param file_path: File path.
    :type file_path: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import silent_remove
        silent_remove('/home/admin/work/projects/examples/1.jpeg')
    """
    try:
        os.remove(file_path)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise


def get_file_size(path: str) -> int:
    """
    Get the size of a file.

    :param path: File path.
    :type path: str
    :returns: File size in bytes
    :rtype: :class:`int`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_file_size
        file_size = get_file_size('/home/admin/work/projects/examples/1.jpeg') # 161665
    """
    return os.path.getsize(path)


def get_directory_size(dir_path: str) -> int:
    """
    Get the size of a directory.

    :param path: Target directory path.
    :type path: str
    :returns: Directory size in bytes
    :rtype: :class:`int`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_directory_size
        dir_size = get_directory_size('/home/admin/work/projects/examples') # 8574563
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += get_file_size(fp)
    return total_size


def archive_directory(
    dir_: str, tar_path: str, split: Optional[Union[int, str]] = None, chunk_size_mb: int = 50
) -> Union[None, List[str]]:
    """
    Create tar archive from directory and optionally split it into parts of specified size.
    You can adjust the size of the chunk to read from the file, while archiving the file into parts.
    Be careful with this parameter, it can affect the performance of the function.
    When spliting, if the size of split is less than the chunk size, the chunk size will be adjusted to fit the split size.

    :param dir_: Target directory path.
    :type dir_: str
    :param tar_path: Path for output tar archive.
    :type tar_path: str
    :param split: Split archive into parts of specified size (in bytes) or size with
        suffix (e.g. '1Kb' = 1024, '1Mb' = 1024 * 1024). Default is None.
    :type split: Union[int, str]
    :param chunk_size_mb: Size of the chunk to read from the file. Default is 50Mb.
    :type chunk_size_mb: int
    :returns: None or list of archive parts if split is not None
    :rtype: Union[None, List[str]]
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import archive_directory
        # If split is not needed.
        archive_directory('/home/admin/work/projects/examples', '/home/admin/work/examples.tar')

        # If split is specified.
        archive_parts_paths = archive_directory('/home/admin/work/projects/examples', '/home/admin/work/examples/archive.tar', split=1000000)
        print(archive_parts_paths) # ['/home/admin/work/examples/archive.tar.001', '/home/admin/work/examples/archive.tar.002']
    """
    with tarfile.open(tar_path, "w", encoding="utf-8") as tar:
        tar.add(dir_, arcname=os.path.sep)

    if split is None:
        return

    split = string_to_byte_size(split)

    if os.path.getsize(tar_path) <= split:
        return

    chunk = chunk_size_mb * 1024 * 1024
    tar_name = os.path.basename(tar_path)
    tar_dir = os.path.abspath(os.path.dirname(tar_path))
    parts_paths = []
    part_number = 1

    original_chunk = chunk
    chunk = min(chunk, split)  # chunk size should be less than split size
    if chunk != original_chunk:
        logger.info(f"Chunk size adjusted to {chunk} bytes to fit split size.")

    with open(tar_path, "rb") as input_file:
        while True:
            part_name = f"{tar_name}.{str(part_number).zfill(3)}"
            output_path = os.path.join(tar_dir, part_name)
            with open(output_path, "wb") as output_file:
                while output_file.tell() < split:
                    data = input_file.read(chunk)
                    if not data:
                        break
                    output_file.write(data)
                if not data and output_file.tell() == 0:
                    os.remove(output_path)
                    break
                parts_paths.append(output_path)
                part_number += 1
            if not data:
                break

    os.remove(tar_path)
    return parts_paths


def unpack_archive(
    archive_path: str, target_dir: str, remove_junk=True, is_split=False, chunk_size_mb: int = 50
) -> None:
    """
    Unpacks archive to the target directory, removes junk files and directories.
    To extract a split archive, you must pass the path to the first part in archive_path. Archive parts must be in the same directory. Format: archive_name.tar.001, archive_name.tar.002, etc. Works with tar and zip.
    You can adjust the size of the chunk to read from the file, while unpacking the file from parts.
    Be careful with this parameter, it can affect the performance of the function.

    :param archive_path: Path to the archive.
    :type archive_path: str
    :param target_dir: Path to the target directory.
    :type target_dir: str
    :param remove_junk: Remove junk files and directories. Default is True.
    :type remove_junk: bool
    :param is_split: Determines if the source archive is split into parts. If True, archive_path must be the path to the first part. Default is False.
    :type is_split: bool
    :param chunk_size_mb: Size of the chunk to read from the file. Default is 50Mb.
    :type chunk_size_mb: int
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        archive_path = '/home/admin/work/examples.tar'
        target_dir = '/home/admin/work/projects'

        sly.fs.unpack_archive(archive_path, target_dir)
    """

    if is_split:
        chunk = chunk_size_mb * 1024 * 1024
        base_name = get_file_name(archive_path)
        dir_name = os.path.dirname(archive_path)
        if get_file_ext(base_name) in (".zip", ".tar"):
            ext = get_file_ext(base_name)
            base_name = get_file_name(base_name)
        else:
            ext = get_file_ext(archive_path)
        parts = sorted([f for f in os.listdir(dir_name) if f.startswith(base_name)])
        combined = os.path.join(dir_name, f"combined{ext}")

        with open(combined, "wb") as output_file:
            for part in parts:
                part_path = os.path.join(dir_name, part)
                with open(part_path, "rb") as input_file:
                    while True:
                        data = input_file.read(chunk)
                        if not data:
                            break
                        output_file.write(data)
        archive_path = combined

    shutil.unpack_archive(archive_path, target_dir)
    if is_split:
        silent_remove(archive_path)
    if remove_junk:
        remove_junk_from_dir(target_dir)


def string_to_byte_size(string: Union[str, int]) -> int:
    """Returns integer representation of byte size from string representation.
        If input is integer, returns the same integer for convenience.

        :param string: string representation of byte size (e.g. 1.5Kb, 2Mb, 3.7Gb, 4.2Tb) or integer
        :type string: Union[str, int]
        :return: integer representation of byte size (or the same integer if input is integer)
        :rtype: int

        :raises ValueError: if input string is invalid

    :Usage example:

    .. code-block:: python
        string_size = "1.5M"
        size = string_to_byte_size(string_size)
        print(size)  # 1572864

    """

    MULTIPLIER = 1024
    units = {"KB": 1, "MB": 2, "GB": 3, "TB": 4}

    if isinstance(string, int):
        return string

    try:
        value, unit = string[:-2], string[-2:].upper()
        multiplier = MULTIPLIER ** units[unit]
        return int(float(value) * multiplier)
    except (KeyError, ValueError, IndexError):
        raise ValueError(
            "Invalid input string. The string must be in the format of '1.5Kb', '2Mb', '3.7Gb', '4.2Tb' or integer."
        )


def get_file_hash(path: str) -> str:
    """
    Get hash from target file.

    :param path: Target file path.
    :type path: str
    :returns: File hash
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import get_file_hash
        hash = get_file_hash('/home/admin/work/projects/examples/1.jpeg') # rKLYA/p/P64dzidaQ/G7itxIz3ZCVnyUhEE9fSMGxU4=
    """
    with open(path, "rb") as file:
        file_bytes = file.read()
        return get_bytes_hash(file_bytes)


def get_file_hash_chunked(path: str, chunk_size: Optional[int] = 1024 * 1024) -> str:
    """
    Get hash from target file by reading it in chunks.

    :param path: Target file path.
    :type path: str
    :param chunk_size: Number of bytes to read per iteration. Default is 1 MB.
    :type chunk_size: int, optional
    :returns: File hash as a base64 encoded string.
    :rtype: str

    :Usage example:

    .. code-block:: python

       file_hash = sly.fs.get_file_hash_chunked('/home/admin/work/projects/examples/1.jpeg')
       print(file_hash)  # Example output: rKLYA/p/P64dzidaQ/G7itxIz3ZCVnyUhEE9fSMGxU4=
    """
    hash_sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_sha256.update(chunk)
    digest = hash_sha256.digest()
    return base64.b64encode(digest).decode("utf-8")


def tree(dir_path: str) -> str:
    """
    Get tree for target directory.

    :param dir_path: Target directory path.
    :type dir_path: str
    :returns: Tree with directory files and subdirectories
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import tree
        dir_tree = tree('/home/admin/work/projects/examples')
        print(dir_tree)
        # Output: /home/admin/work/projects/examples
        # ├── [4.0K]  1
        # │   ├── [165K]  crop.jpeg
        # │   ├── [169K]  fliplr.jpeg
        # │   ├── [169K]  flipud.jpeg
        # │   ├── [166K]  relative_crop.jpeg
        # │   ├── [167K]  resize.jpeg
        # │   ├── [169K]  rotate.jpeg
        # │   ├── [171K]  scale.jpeg
        # │   └── [168K]  translate.jpeg
        # ├── [ 15K]  123.jpeg
        # ├── [158K]  1.jpeg
        # ├── [188K]  1.txt
        # ├── [1.3M]  1.zip
        # ├── [4.0K]  2
        # ├── [ 92K]  acura.png
        # ├── [1.2M]  acura_PNG122.png
        # ├── [198K]  aston_martin_PNG55.png
        # ├── [4.0K]  ds1
        # │   ├── [4.0K]  ann
        # │   │   ├── [4.3K]  IMG_0748.jpeg.json
        # │   │   ├── [ 151]  IMG_0777.jpeg.json
        # │   │   ├── [ 151]  IMG_0888.jpeg.json
        # │   │   ├── [3.7K]  IMG_1836.jpeg.json
        # │   │   ├── [8.1K]  IMG_2084.jpeg.json
        # │   │   ├── [5.5K]  IMG_3861.jpeg.json
        # │   │   ├── [6.0K]  IMG_4451.jpeg.json
        # │   │   └── [5.0K]  IMG_8144.jpeg.json
        # │   └── [4.0K]  img
        # │       ├── [152K]  IMG_0748.jpeg
        # │       ├── [210K]  IMG_0777.jpeg
        # │       ├── [210K]  IMG_0888.jpeg
        # │       ├── [137K]  IMG_1836.jpeg
        # │       ├── [139K]  IMG_2084.jpeg
        # │       ├── [145K]  IMG_3861.jpeg
        # │       ├── [133K]  IMG_4451.jpeg
        # │       └── [136K]  IMG_8144.jpeg
        # ├── [152K]  example.jpeg
        # ├── [2.4K]  example.json
        # ├── [153K]  flip.jpeg
        # ├── [ 65K]  hash1.jpeg
        # ├── [ 336]  meta.json
        # └── [5.4K]  q.jpeg
        # 5 directories, 37 files
    """
    out = subprocess.Popen(
        ["tree", "--filelimit", "500", "-h", "-n", dir_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, stderr = out.communicate()
    return stdout.decode("utf-8")


def log_tree(
    dir_path: str,
    logger,
    level: Literal["info", "debug", "warning", "error"] = "info",
) -> None:
    """
    Get tree for target directory and displays it in the log.

    :param dir_path: Target directory path.
    :type dir_path: str
    :param logger: Logger to display data.
    :type logger: logger
    :type level: Logger level. Available levels: info, debug, warning, error. Default: info.
    :type level: Literal["info", "debug", "warning", "error"]
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import log_tree
        logger = sly.logger
        log_tree('/home/admin/work/projects/examples', logger)
    """
    out = tree(dir_path)

    log_levels = {
        "info": logger.info,
        "debug": logger.debug,
        "warning": logger.warning,
        "error": logger.error,
    }
    if level not in log_levels:
        raise ValueError(
            f"Unknown logger level: {level}. Available levels: info, debug, warning, error"
        )
    log_func = log_levels[level]
    log_func("DIRECTORY_TREE", extra={"tree": out})


def touch(path: str) -> None:
    """
    Sets access and modification times for a file.

    :param path: Target file path.
    :type path: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import touch
        touch('/home/admin/work/projects/examples/1.jpeg')
    """
    ensure_base_path(path)
    with open(path, "a"):
        os.utime(path, None)


def download(
    url: str,
    save_path: str,
    cache: Optional[FileCache] = None,
    progress: Optional[Callable] = None,
    headers: Optional[Dict] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Load image from url to host by target path.

    :param url: Target file path.
    :type url: str
    :param url: The path where the file is saved.
    :type url: str
    :param cache: An instance of FileCache class that provides caching functionality for the downloaded content. If None, caching is disabled.
    :type cache: FileCache, optional
    :param progress: Function for tracking download progress.
    :type progress: Progress, optional
    :param headers: A dictionary of HTTP headers to include in the request.
    :type headers: Dict, optional.
    :param timeout: The maximum number of seconds to wait for a response from the server. If the server does not respond within the timeout period, a TimeoutError is raised.
    :type timeout: int, optional.
    :returns: Full path to downloaded image
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        from supervisely.io.fs import download
        img_link = 'https://m.media-amazon.com/images/M/MV5BMTYwOTEwNjAzMl5BMl5BanBnXkFtZTcwODc5MTUwMw@@._V1_.jpg'
        im_path = download(img_link, '/home/admin/work/projects/examples/avatar.jpeg')
        print(im_path)
        # Output:
        # /home/admin/work/projects/examples/avatar.jpeg

        # if you need to specify some headers
        headers = {
            'User-Agent': 'Mozilla/5.0',
        }
        im_path = download(img_link, '/home/admin/work/projects/examples/avatar.jpeg', headers=headers)
        print(im_path)
        # Output:
        # /home/admin/work/projects/examples/avatar.jpeg

    """

    def _download():
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                total_size_in_bytes = int(CaseInsensitiveDict(r.headers).get("Content-Length", "0"))
                if progress is not None and type(progress) is Progress:
                    progress.set(0, total_size_in_bytes)
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        if progress is not None:
                            if type(progress) is Progress:
                                progress.iters_done_report(len(chunk))
                            else:
                                progress(len(chunk))
        except requests.exceptions.Timeout as e:
            message = (
                "Request timed out. "
                "This may be due to server-side security measures, network congestion, or other issues. "
                "Please check your server logs for more information."
            )
            logger.warn(msg=message)
            raise e

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
                sizeb = get_file_size(save_path)
                if type(progress) is Progress:
                    progress.set(0, sizeb)
                    progress.iters_done_report(sizeb)
                else:
                    progress(sizeb)

    return save_path


def copy_dir_recursively(
    src_dir: str, dst_dir: str, progress_cb: Optional[Union[tqdm, Callable]] = None
) -> List[str]:
    mkdir(dst_dir)
    src_dir_norm = src_dir.rstrip(os.sep)

    for rel_sub_dir in get_subdirs(src_dir_norm, recursive=True):
        dst_sub_dir = os.path.join(dst_dir, rel_sub_dir)
        mkdir(dst_sub_dir)

    files = list_files_recursively(src_dir_norm)
    for src_file_path in files:
        dst_file_path = os.path.normpath(src_file_path.replace(src_dir_norm, dst_dir))
        ensure_base_path(dst_file_path)
        if not file_exists(dst_file_path):
            copy_file(src_file_path, dst_file_path)
            if progress_cb is not None:
                progress_cb(get_file_size(src_file_path))


def is_on_agent(remote_path: str) -> bool:
    """Check if remote_path starts is on agent (e.g. starts with 'agent://<agent-id>/').

    :param remote_path: path to check
    :type remote_path: str
    :return: True if remote_path starts with 'agent://<agent-id>/' and False otherwise
    :rtype: bool
    """
    if remote_path.startswith("agent://"):
        return True
    else:
        return False


def parse_agent_id_and_path(remote_path: str) -> Tuple[int, str]:
    """Return agent id and path in agent folder from remote_path.

    :param remote_path: path to parse
    :type remote_path: str
    :return: agent id and path in agent folder
    :rtype: Tuple[int, str]
    :raises ValueError: if remote_path doesn't start with 'agent://<agent-id>/'
    :Usage example:

     .. code-block:: python

        import os
        from dotenv import load_dotenv

        import supervisely as sly

        # Load secrets and create API object from .env file (recommended)
        # Learn more here: https://developer.supervisely.com/getting-started/basics-of-authentication
        load_dotenv(os.path.expanduser("~/supervisely.env"))
        api = sly.Api.from_env()

        # Parse agent id and path in agent folder from remote_path
        remote_path = "agent://1/agent_folder/subfolder/file.txt"
        agent_id, path_in_agent_folder = sly.fs.parse_agent_id_and_path(remote_path)
        print(agent_id)  # 1
        print(path_in_agent_folder)  # /agent_folder/subfolder/file.txt
    """
    if is_on_agent(remote_path) is False:
        raise ValueError("agent path have to starts with 'agent://<agent-id>/'")
    search = re.search("agent://(\d+)(.*)", remote_path)
    agent_id = int(search.group(1))
    path_in_agent_folder = search.group(2)
    if not path_in_agent_folder.startswith("/"):
        path_in_agent_folder = "/" + path_in_agent_folder
    if remote_path.endswith("/") and not path_in_agent_folder.endswith("/"):
        path_in_agent_folder += "/"
    # path_in_agent_folder = os.path.normpath(path_in_agent_folder)
    return agent_id, path_in_agent_folder


def dirs_with_marker(
    input_path: str,
    markers: Union[str, List[str]],
    check_function: Callable = None,
    ignore_case: bool = False,
) -> Generator[str, None, None]:
    """
    Generator that yields paths to directories that contain markers files. If the check_function is specified,
    then the markered directory will be yielded only if the check_function returns True. The check_function
    must take a single argument - the path to the markered directory and return True or False.

    :param input_path: path to the directory in which the search will be performed
    :type input_path: str
    :param markers: single marker or list of markers (e.g. 'config.json' or ['config.json', 'config.yaml'])
    :type markers: Union[str, List[str]]
    :param check_function: function to check that directory meets the requirements and returns bool
    :type check_function: Callable
    :param ignore_case: ignore case when searching for markers
    :type ignore_case: bool
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        input_path = '/home/admin/work/projects/examples'

        # You can pass a string if you have only one marker.
        # markers = 'config.json'

        # Or a list of strings if you have several markers.
        # There's no need to pass one marker in different cases, you can use ignore_case=True for this.
        markers = ['config.json', 'config.yaml']


        # Check function is optional, if you don't need the directories to meet any requirements,
        # you can omit it.

        def check_function(dir_path):
            test_file_path = os.path.join(dir_path, 'test.txt')
            return os.path.exists(test_file_path)

        for directory in sly.fs.dirs_with_marker(input_path, markers, check_function, ignore_case=True):
            # Now you can be sure that the directory contains the markers and meets the requirements.
            # Do something with it.
            print(directory)
    """

    if isinstance(markers, str):
        markers = [markers]

    paths = list_dir_recursively(input_path)
    for path in paths:
        for marker in markers:
            filename = get_file_name_with_ext(path)
            if ignore_case:
                filename = filename.lower()
                marker = marker.lower()

            if filename == marker:
                parent_dir = os.path.dirname(path)
                markered_dir = os.path.join(input_path, parent_dir)

                if check_function is None or check_function(markered_dir):
                    yield markered_dir


def dirs_filter(input_path: str, check_function: Callable) -> Generator[str, None, None]:
    """
    Generator that yields paths to directories that meet the requirements of the check_function.

    :param input_path: path to the directory in which the search will be performed
    :type input_path: str
    :param check_function: function to check that directory meets the requirements and returns bool
    :type check_function: Callable
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        input_path = '/home/admin/work/projects/examples'

        # Prepare the check function.

        def check_function(directory) -> bool:
            images_dir = os.path.join(directory, "images")
            annotations_dir = os.path.join(directory, "annotations")

            return os.path.isdir(images_dir) and os.path.isdir(annotations_dir)

        for directory in sly.fs.dirs(input_path, check_function):
            # Now you can be sure that the directory meets the requirements.
            # Do something with it.
            print(directory)
    """
    paths = [os.path.abspath(input_path)]
    paths.extend(list_dir_recursively(input_path, include_subdirs=True, use_global_paths=True))
    for path in paths:
        if os.path.isdir(path):
            if check_function(path):
                yield path


def change_directory_at_index(path: str, dir_name: str, dir_index: int) -> str:
    """
    Change directory name in path by index.
    If you use counting from the end, keep in mind that if the path ends with a file, the file will be assigned to the last index.

    :param path: The original path
    :type path: str
    :param dir_name: Directory name
    :type dir_name: str
    :param dir_index: Index of the directory we want to change, negative values count from the end
    :type dir_index: int
    :return: New path
    :rtype: str
    :raises IndexError: If the catalog index is out of bounds for a given path
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        input_path = 'head/dir_1/file.txt'
        new_path = sly.io.fs.change_directory_at_index(input_path, 'dir_2', -2)

        print(new_path)

    """
    path_components = path.split(os.path.sep)
    if -len(path_components) <= dir_index < len(path_components):
        path_components[dir_index] = dir_name
    else:
        raise IndexError(
            f"Path index '{dir_index}' is out of bounds 'path_components={len(path_components)}' for a given path"
        )
    return os.path.sep.join(path_components)


def is_archive(file_path: str) -> bool:
    """
    Checks if the file is an archive by its mimetype using list of the most common archive mimetypes.

    :param local_path: path to the local file
    :type local_path: str
    :return: True if the file is an archive, False otherwise
    :rtype: bool
    """
    archive_mimetypes = [
        "application/zip",
        "application/x-tar",
        "application/x-gzip",
        "application/x-bzip2",
        "application/x-7z-compressed",
        "application/x-rar-compressed",
        "application/x-xz",
        "application/x-lzip",
        "application/x-lzma",
        "application/x-lzop",
        "application/x-bzip",
        "application/x-bzip2",
        "application/x-compress",
        "application/x-compressed",
    ]

    return mimetypes.guess_type(file_path)[0] in archive_mimetypes


def str_is_url(string: str) -> bool:
    """
    Check if string is a valid URL.
    :param string: string to check
    :type string: str
    :return: True if string is a valid URL, False otherwise
    :rtype: bool
    :Usage example:
     .. code-block:: python
        import supervisely as sly
        url = 'https://example.com/image.jpg'
        is_url = sly.fs.str_is_url(url)
        print(is_url)  # True
    """
    from urllib.parse import urlparse

    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


async def copy_file_async(
    src: str,
    dst: str,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    progress_cb_type: Literal["number", "size"] = "size",
) -> None:
    """
    Asynchronously copy file from one path to another, if destination directory doesn't exist it will be created.

    :param src: Source file path.
    :type src: str
    :param dst: Destination file path.
    :type dst: str
    :param progress_cb: Function for tracking copy progress.
    :type progress_cb: Union[tqdm, Callable], optional
    :param progress_cb_type: Type of progress callback. Can be "number" or "size". Default is "size".
    :type progress_cb_type: Literal["number", "size"], optional
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely._utils import run_coroutine

        coroutine = sly.fs.copy_file_async('/home/admin/work/projects/example/1.png', '/home/admin/work/tests/2.png')
        run_coroutine(coroutine)
    """
    ensure_base_path(dst)
    async with aiofiles.open(dst, "wb") as out_f:
        async with aiofiles.open(src, "rb") as in_f:
            while True:
                chunk = await in_f.read(1024 * 1024)
                if not chunk:
                    break
                await out_f.write(chunk)
                if progress_cb is not None and progress_cb_type == "size":
                    progress_cb(len(chunk))
    if progress_cb is not None and progress_cb_type == "number":
        progress_cb(1)


async def get_file_hash_async(path: str) -> str:
    """
    Get hash from target file asynchronously.

    :param path: Target file path.
    :type path: str
    :returns: File hash
    :rtype: :class:`str`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely._utils import run_coroutine

        coroutine = sly.fs.get_file_hash_async('/home/admin/work/projects/examples/1.jpeg')
        hash = run_coroutine(coroutine)
    """
    async with aiofiles.open(path, "rb") as file:
        file_bytes = await file.read()
        return get_bytes_hash(file_bytes)


async def unpack_archive_async(
    archive_path: str, target_dir: str, remove_junk=True, is_split=False, chunk_size_mb: int = 50
) -> None:
    """
    Unpacks archive to the target directory, removes junk files and directories.
    To extract a split archive, you must pass the path to the first part in archive_path. Archive parts must be in the same directory. Format: archive_name.tar.001, archive_name.tar.002, etc. Works with tar and zip.
    You can adjust the size of the chunk to read from the file, while unpacking the file from parts.
    Be careful with this parameter, it can affect the performance of the function.

    :param archive_path: Path to the archive.
    :type archive_path: str
    :param target_dir: Path to the target directory.
    :type target_dir: str
    :param remove_junk: Remove junk files and directories. Default is True.
    :type remove_junk: bool
    :param is_split: Determines if the source archive is split into parts. If True, archive_path must be the path to the first part. Default is False.
    :type is_split: bool
    :param chunk_size_mb: Size of the chunk to read from the file. Default is 50Mb.
    :type chunk_size_mb: int
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely._utils import run_coroutine

        archive_path = '/home/admin/work/examples.tar'
        target_dir = '/home/admin/work/projects'

        coroutine = sly.fs.unpack_archive_async(archive_path, target_dir)
        run_coroutine(coroutine)
    """
    if is_split:
        chunk = chunk_size_mb * 1024 * 1024
        base_name = get_file_name(archive_path)
        dir_name = os.path.dirname(archive_path)
        if get_file_ext(base_name) in (".zip", ".tar"):
            ext = get_file_ext(base_name)
            base_name = get_file_name(base_name)
        else:
            ext = get_file_ext(archive_path)
        parts = sorted([f for f in os.listdir(dir_name) if f.startswith(base_name)])
        combined = os.path.join(dir_name, f"combined{ext}")

        async with aiofiles.open(combined, "wb") as output_file:
            for part in parts:
                part_path = os.path.join(dir_name, part)
                async with aiofiles.open(part_path, "rb") as input_file:
                    while True:
                        data = await input_file.read(chunk)
                        if not data:
                            break
                        await output_file.write(data)
        archive_path = combined

    loop = get_or_create_event_loop()
    await loop.run_in_executor(None, shutil.unpack_archive, archive_path, target_dir)
    if is_split:
        silent_remove(archive_path)
    if remove_junk:
        remove_junk_from_dir(target_dir)


async def touch_async(path: str) -> None:
    """
    Sets access and modification times for a file asynchronously.

    :param path: Target file path.
    :type path: str
    :returns: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        from supervisely._utils import run_coroutine

        coroutine = sly.fs.touch_async('/home/admin/work/projects/examples/1.jpeg')
        run_coroutine(coroutine)
    """
    ensure_base_path(path)
    async with aiofiles.open(path, "a"):
        loop = get_or_create_event_loop()
        await loop.run_in_executor(None, os.utime, path, None)


async def list_files_recursively_async(
    dir_path: str,
    valid_extensions: Optional[List[str]] = None,
    filter_fn: Optional[Callable[[str], bool]] = None,
    ignore_valid_extensions_case: bool = False,
) -> List[str]:
    """
    Recursively list files in the directory asynchronously.
    Returns list with all file paths.
    Can be filtered by valid extensions and filter function.

    :param dir_path: Target directory path.
    :type dir_path: str
    :param valid_extensions: List of valid extensions. Default is None.
    :type valid_extensions: Optional[List[str]]
    :param filter_fn: Filter function. Default is None.
    :type filter_fn: Optional[Callable[[str], bool]]
    :param ignore_valid_extensions_case: Ignore case when checking valid extensions. Default is False.
    :type ignore_valid_extensions_case: bool
    :returns: List of file paths
    :rtype: List[str]

    :Usage example:

         .. code-block:: python

            import supervisely as sly
            from supervisely._utils import run_coroutine

            dir_path = '/home/admin/work/projects/examples'

            coroutine = sly.fs.list_files_recursively_async(dir_path)
            files = run_coroutine(coroutine)
    """

    def sync_file_list():
        if valid_extensions and ignore_valid_extensions_case:
            valid_extensions_set = set(map(str.lower, valid_extensions))
        else:
            valid_extensions_set = set(valid_extensions) if valid_extensions else None

        files = []
        for dir_name, _, file_names in os.walk(dir_path):
            full_paths = [os.path.join(dir_name, filename) for filename in file_names]

            if valid_extensions_set:
                full_paths = [
                    fp
                    for fp in full_paths
                    if (
                        ext := (
                            os.path.splitext(fp)[1].lower()
                            if ignore_valid_extensions_case
                            else os.path.splitext(fp)[1]
                        )
                    )
                    in valid_extensions_set
                ]

            if filter_fn:
                full_paths = [fp for fp in full_paths if filter_fn(fp)]

            files.extend(full_paths)

        return files

    loop = get_or_create_event_loop()
    return await loop.run_in_executor(None, sync_file_list)


def get_file_offsets_batch_generator(
    archive_path: str,
    team_file_id: Optional[int] = None,
    filter_func: Optional[Callable] = None,
    output_format: Literal["dicts", "objects"] = "dicts",
    batch_size: int = OFFSETS_PKL_BATCH_SIZE,
) -> Generator[Union[List[Dict], List["BlobImageInfo"]], None, None]:
    """
    Extracts offset information for files from TAR archives and returns a generator that yields the information in batches.

    `team_file_id` may be None if it's not possible to obtain the ID at this moment.
    You can set the `team_file_id` later when uploading the file to Supervisely.

    :param archive_path: Local path to the archive
    :type archive_path: str
    :param team_file_id: ID of file in Team Files. Default is None.
                    `team_file_id` may be None if it's not possible to obtain the ID at this moment.
                    You can set the `team_file_id` later when uploading the file to Supervisely.
    :type team_file_id: Optional[int]
    :param filter_func: Function to filter files. The function should take a filename as input and return True if the file should be included.
    :type filter_func: Callable, optional
    :param output_format: Format of the output. Default is `dicts`.
                   `objects` - returns a list of BlobImageInfo objects.
                   `dicts` - returns a list of dictionaries.
    :type output_format: Literal["dicts", "objects"]
    :returns: Generator yielding batches of file information in the specified format.
    :rtype: Generator[Union[List[Dict], List[BlobImageInfo]]], None, None]

    :raises ValueError: If the archive type is not supported or contains compressed files
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        archive_path = '/home/admin/work/projects/examples.tar'
        file_infos = sly.fs.get_file_offsets_batch_generator(archive_path)
        for batch in file_infos:
            print(batch)

        # Output:
        # [
        #     {
        #         "title": "image1.jpg",
        #         "teamFileId": None,
        #         "sourceBlob": {
        #             "offsetStart": 0,
        #             "offsetEnd": 123456
        #         }
        #     },
        #     {
        #         "title": "image2.jpg",
        #         "teamFileId": None,
        #         "sourceBlob": {
        #             "offsetStart": 123456,
        #             "offsetEnd": 234567
        #         }
        #     }
        # ]
    """
    from supervisely.api.image_api import BlobImageInfo

    ext = Path(archive_path).suffix.lower()

    if ext == ".tar":
        if output_format == "dicts":
            yield from _process_tar_generator(
                tar_path=archive_path,
                team_file_id=team_file_id,
                filter_func=filter_func,
                batch_size=batch_size,
            )
        else:
            for batch in _process_tar_generator(
                tar_path=archive_path,
                team_file_id=team_file_id,
                filter_func=filter_func,
                batch_size=batch_size,
            ):
                blob_file_infos = [BlobImageInfo.from_dict(file_info) for file_info in batch]
                yield blob_file_infos
    else:
        raise ValueError(f"Unsupported archive type: {ext}. Only .tar are supported")


def _process_tar_generator(
    tar_path: str,
    team_file_id: Optional[int] = None,
    filter_func: Optional[Callable] = None,
    batch_size: int = OFFSETS_PKL_BATCH_SIZE,
) -> Generator[List[Dict], None, None]:
    """
    Processes a TAR archive and yields batches of offset information for files.

    :param tar_path: Path to the TAR archive
    :type tar_path: str
    :param team_file_id: ID of the team file, defaults to None
    :type team_file_id: Optional[int], optional
    :param filter_func: Function to filter files. The function should take a filename as input and return True if the file should be included.
    :type filter_func: Optional[Callable], optional
    :param batch_size: Number of files in each batch, defaults to 10000
    :type batch_size: int, optional
    :yield: Batches of dictionaries with file offset information
    :rtype: Generator[List[Dict], None, None]
    """
    from supervisely.api.api import ApiField

    with tarfile.open(tar_path, "r") as tar:
        batch = []
        processed_count = 0
        members = tar.getmembers()
        total_members_count = len(members)  # for logging

        logger.debug(f"Processing TAR archive with {total_members_count} members")

        for member in members:
            skip = not member.isfile()

            if filter_func and not filter_func(member.name):
                logger.debug(f"File '{member.name}' is skipped by filter function")
                skip = True

            if not skip:
                file_info = {
                    ApiField.TITLE: os.path.basename(member.name),
                    ApiField.TEAM_FILE_ID: team_file_id,
                    ApiField.SOURCE_BLOB: {
                        ApiField.OFFSET_START: member.offset_data,
                        ApiField.OFFSET_END: member.offset_data + member.size,
                    },
                }
                batch.append(file_info)

                # Yield batch when it reaches the specified size
                if len(batch) >= batch_size:
                    processed_count += len(batch)
                    logger.debug(
                        f"Yielding batch of {len(batch)} files, processed {processed_count} files so far"
                    )
                    yield batch
                    batch = []

        # Yield any remaining files in the last batch
        if batch:
            processed_count += len(batch)
            logger.debug(
                f"Yielding final batch of {len(batch)} files, processed {processed_count} files total"
            )
            yield batch


def save_blob_offsets_pkl(
    blob_file_path: str,
    output_dir: str,
    team_file_id: Optional[int] = None,
    filter_func: Optional[Callable] = None,
    batch_size: int = OFFSETS_PKL_BATCH_SIZE,
    replace: bool = False,
) -> str:
    """
    Processes blob file locally and creates a pickle file with offset information.

    :param blob_file_path: Path to the local blob file
    :type blob_file_path: str
    :param output_dir: Path to the output directory
    :type output_dir: str
    :param team_file_id: ID of file in Team Files. Default is None.
                    `team_file_id` may be None if it's not possible to obtain the ID at this moment.
                    You can set the `team_file_id` later when uploading the file to Supervisely.
    :type team_file_id: Optional[int]
    :param filter_func: Function to filter files. The function should take a filename as input and return True if the file should be included.
    :type filter_func: Callable, optional
    :param batch_size: Number of files to process in each batch, defaults to 10000
    :type batch_size: int, optional
    :param replace: If True, overwrite the existing file if it exists.
                    If False, skip processing if the file already exists and return its path.
                    Default is False.
    :type replace: bool
    :returns: Path to the output pickle file
    :rtype: str

    :Usage example:

        .. code-block:: python

            import supervisely as sly

            archive_path = '/path/to/examples.tar'
            output_dir = '/path/to/output'
            sly.fs.save_blob_offsets_pkl(archive_path, output_dir)
    """
    from supervisely.api.image_api import BlobImageInfo

    archive_name = Path(blob_file_path).stem
    output_path = os.path.join(output_dir, archive_name + OFFSETS_PKL_SUFFIX)

    if file_exists(output_path):
        logger.debug(f"Offsets file already exists: {output_path}")
        if replace:
            logger.debug(f"Replacing existing offsets file: {output_path}")
            silent_remove(output_path)
        else:
            logger.debug(f"Skipping processing, using existing offsets file: {output_path}")
            return output_path

    offsets_batch_generator = get_file_offsets_batch_generator(
        archive_path=blob_file_path,
        team_file_id=team_file_id,
        filter_func=filter_func,
        output_format="objects",
        batch_size=batch_size,
    )

    BlobImageInfo.dump_to_pickle(offsets_batch_generator, output_path)
    return output_path
