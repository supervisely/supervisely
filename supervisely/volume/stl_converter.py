import os
import shutil
import glob
import nrrd
import math
import supervisely
import trimesh
import numpy as np

from typing import List, Dict, Any
from supervisely import logger
from supervisely.io.fs import get_file_name_with_ext, file_exists, dir_exists, mkdir

_stl_ext = ".stl"
_nrrd_ext = ".nrrd"
volume_dir_name = "volume"
annotation_dir_name = "ann"
interpolation_dir_name = "interpolation"
old_interpolation_dir_name = "old_interpolation"


def matrix_from_nrrd_header(header: Dict) -> np.ndarray:
    """
    Creates matrix from NRRD header

    :param header: Dictionary containing the header fields and their corresponding parsed value
    :type header: :class:`dict` (:class:`str`, :obj:`Object`)
    :return: Transformation matrix that maps voxel coordinates to world coordinates
    :rtype: np.ndarray
    :raises IOError: Need "{}" field of header to determine the mapping from voxels to world coordinates
    """
    try:
        space_directions = header["space directions"]
        space_origin = header["space origin"]
    except KeyError as e:
        raise IOError(
            'Need the header\'s "{}" field to determine the mapping from voxels to world coordinates.'.format(
                e
            )
        )

    # "... the space directions field gives, one column at a time, the mapping from image space to world space
    # coordinates ... [1]_" -> list of columns, needs to be transposed
    trans_3x3 = np.array(space_directions).T
    trans_4x4 = np.eye(4)
    trans_4x4[:3, :3] = trans_3x3
    trans_4x4[:3, 3] = space_origin

    return trans_4x4


def voxels_to_mask(mask_shape: List, voxel_to_world: np.ndarray, stl_path: str) -> np.ndarray:
    """
    Converts STL voxels to NRRD mask

    :param mask_shape: Value "sizes" of Volume nrrd_header
    :type mask_shape: List
    :param voxel_to_world: Transformation matrix that maps voxel coordinates to world coordinates
    :type voxel_to_world: np.ndarray
    :param stl_path: Path to STL file
    :type stl_path: str
    :return: Mask
    :rtype: np.ndarray
    """

    world_to_voxel = np.linalg.inv(voxel_to_world)

    mesh = trimesh.load(stl_path)

    min_vec = [float("inf"), float("inf"), float("inf")]
    max_vec = [float("-inf"), float("-inf"), float("-inf")]

    mesh.apply_scale((-1, -1, 1))  # LPS to RAS
    mesh.apply_transform(world_to_voxel)

    for vert in mesh.vertices:
        min_vec[0] = min(min_vec[0], vert[0])
        min_vec[1] = min(min_vec[1], vert[1])
        min_vec[2] = min(min_vec[2], vert[2])

        max_vec[0] = max(max_vec[0], vert[0])
        max_vec[1] = max(max_vec[1], vert[1])
        max_vec[2] = max(max_vec[2], vert[2])

    center = [(min_v + max_v) / 2 for min_v, max_v in zip(min_vec, max_vec)]

    try:
        voxel = mesh.voxelized(pitch=1.0)
    except Exception as e:
        supervisely.logger.error(e)
        supervisely.logger.warning(
            "Couldn't voxelize file {!r}".format(get_file_name_with_ext(stl_path)),
            extra={"file_path": stl_path},
        )
        return np.zeros(mask_shape).astype(np.bool)

    voxel = voxel.fill()
    mask = voxel.matrix.astype(np.bool)
    padded_mask = np.zeros(mask_shape).astype(np.bool)

    # find dimension coords
    start = [math.ceil(center_v - shape_v / 2) for center_v, shape_v in zip(center, mask.shape)]
    end = [math.ceil(center_v + shape_v / 2) for center_v, shape_v in zip(center, mask.shape)]

    # find intersections
    vol_inter_max = [max(start[0], 0), max(start[1], 0), max(start[2], 0)]
    vol_inter_min = [
        min(end[0], mask_shape[0]),
        min(end[1], mask_shape[1]),
        min(end[2], mask_shape[2]),
    ]

    padded_mask[
        vol_inter_max[0] : vol_inter_min[0],
        vol_inter_max[1] : vol_inter_min[1],
        vol_inter_max[2] : vol_inter_min[2],
    ] = mask[
        vol_inter_max[0] - start[0] : vol_inter_min[0] - start[0],
        vol_inter_max[1] - start[1] : vol_inter_min[1] - start[1],
        vol_inter_max[2] - start[2] : vol_inter_min[2] - start[2],
    ]
    padded_mask = padded_mask.astype(dtype="uint8")
    return padded_mask


def save_to_nrrd_file_on_upload(interpolation_dir: str) -> List[str]:
    """
    Save STL as NRRD containing 3D Mask in the same directory with the same name

    :param interpolation_dir: Path to the directory where the interpolation is stored
    :type interpolation_dir: str
    :return: List of paths where interpolations in NRRD format are stored or None
    :rtype: List[str]
    """
    # additional check for interpolation folder
    if not dir_exists(interpolation_dir):
        return None

    make_warn = False

    files_list = glob.glob(os.path.join(interpolation_dir, "*"))
    nrrd_paths = []

    for file in files_list:
        if os.path.splitext(file)[1] == _stl_ext:
            make_warn = True
            nrrd_path = file.replace(_stl_ext, _nrrd_ext)

            # doesn't need to convert if already exists interpolation in NRRD
            if file_exists(nrrd_path):
                make_warn = False
                continue

            volume_path = os.path.dirname(file.replace(interpolation_dir_name, volume_dir_name))
            _, header = nrrd.read(volume_path)
            world_matrix = matrix_from_nrrd_header(header)
            shape = header["sizes"]

            mask = voxels_to_mask(shape, world_matrix, file)

            nrrd.write(nrrd_path, mask, header)
            nrrd_paths.append(nrrd_path)
    if make_warn:
        logger.warning(
            f"STL format is no longer supported. STL is automatically converted to NRRD containing 3D Mask"
        )
    return nrrd_paths


def save_to_nrrd_file_on_download(mesh_paths: str) -> List[str]:
    """
    Save STL as NRRD containing 3D Mask in the same directory with the same name.
    Move STL file to 'old_interpolation' directory.

    :param mesh_paths: Path to the directory where the interpolation is stored
    :type mesh_paths: str
    :return: List of paths where interpolations in NRRD format are stored or None
    :rtype: List[str]
    """
    # additional check for interpolation folder
    if not mesh_paths:
        return []

    logger.warning(
        f"STL format is no longer supported. STL is automatically converted to NRRD containing 3D Mask and moved to 'old_interpolation' directory"
    )

    nrrd_paths = []

    for file in mesh_paths:
        stl_path = os.path.join(file)
        nrrd_path = stl_path.replace(_stl_ext, _nrrd_ext)
        volume_path = os.path.dirname(stl_path.replace(interpolation_dir_name, volume_dir_name))
        _, header = nrrd.read(volume_path)
        world_matrix = matrix_from_nrrd_header(header)
        shape = header["sizes"]

        mask = voxels_to_mask(shape, world_matrix, stl_path)

        nrrd.write(nrrd_path, mask, header)
        nrrd_paths.append(nrrd_path)

        dir = os.path.split(file)[0].replace(interpolation_dir_name, old_interpolation_dir_name)

        if not dir_exists(dir):
            mkdir(dir)

        shutil.move(stl_path, stl_path.replace(interpolation_dir_name, old_interpolation_dir_name))

    return nrrd_paths
