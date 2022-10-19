# coding: utf-8

import os
import numpy as np
import open3d as o3d
from typing import List, Optional
from supervisely._utils import is_development, abs_url
from supervisely.io.fs import ensure_base_path

# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_POINTCLOUD_EXTENSIONS = [".pcd", ".ply"]


class PointcloudExtensionError(Exception):
    pass


class UnsupportedPointcloudFormat(Exception):
    pass


class PointcloudReadException(Exception):
    pass


def is_valid_ext(ext: str) -> bool:
    """
    Checks if given extention is supported
    :param ext: str
    :return: bool
    """
    return ext.lower() in ALLOWED_POINTCLOUD_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    """
    Checks if file from given path with given extention is supported
    :param path: str
    :return: bool
    """
    return is_valid_ext(os.path.splitext(path)[1])


def validate_ext(ext: str) -> None:
    """
    Raise error if given extention is not supported
    :param ext: str
    """
    if not is_valid_ext(ext):
        raise UnsupportedPointcloudFormat(
            "Unsupported pointcloud extension: {}. Only the following extensions are supported: {}.".format(
                ext, ALLOWED_POINTCLOUD_EXTENSIONS
            )
        )


def validate_format(path):
    _, ext = os.path.splitext(path)
    validate_ext(ext)


def get_labeling_tool_url(dataset_id, pointcloud_id):
    res = f"/app/point-clouds/?datasetId={dataset_id}&pointCloudId={pointcloud_id}"
    if is_development():
        res = abs_url(res)
    return res


def get_labeling_tool_link(url, name="open in labeling tool"):
    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{name}<i class="zmdi zmdi-open-in-new" style="margin-left: 5px"></i></a>'


def read(path: str, coords_dims: Optional[List[int]] = None) -> np.ndarray:
    """
    Loads a pointcloud from the specified file and returns it in XYZ format.

    :param path: Path to file.
    :type path: str
    :return: Numpy array
    :rtype: :class:`np.ndarray`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        ptc = sly.pointcloud.read('/home/admin/work/pointclouds/ptc0.pcd')
    """
    validate_format(path)
    if coords_dims is None:
        coords_dims = [0, 1, 2]
    pcd_data = o3d.io.read_point_cloud(path)
    if pcd_data is None:
        raise IOError(f"open3d can not open the file {path}")
    pointcloud_np = np.asarray(pcd_data.points)
    pointcloud_np = pointcloud_np[:, coords_dims]
    return pointcloud_np


def write(path: str, pointcloud_np: np.ndarray, coords_dims: Optional[List[int]] = None) -> bool:
    """
    Saves a pointcloud to the specified file. It creates directory from path if the directory for this path does not exist.

    :param path: Path to file.
    :type path: str
    :param pointcloud_np: Pointcloud [N, 3] in XYZ format.
    :type pointcloud_np: :class:`np.ndarray`
    :param coords_dims: List of indexes for (X, Y, Z) coords. Default (if None): [0, 1, 2].
    :type coords_dims: Optional[List[int]]
    :return: Success or not.
    :rtype: bool
    :Usage example:

     .. code-block:: python

        import supervisely as sly
        import numpy as np

        pointcloud = np.random.randn(100, 3)

        ptc = sly.pointcloud.write('/home/admin/work/pointclouds/ptc0.pcd', pointcloud)
    """
    ensure_base_path(path)
    validate_format(path)
    if coords_dims is None:
        coords_dims = [0, 1, 2]
    pointcloud_np = pointcloud_np[:, coords_dims]
    pcd_data = o3d.geometry.PointCloud()
    pcd_data.points = o3d.utility.Vector3dVector(pointcloud_np)
    return o3d.io.write_point_cloud(path, pcd_data)
