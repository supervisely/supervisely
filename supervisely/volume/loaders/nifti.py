#!/usr/bin/env python
# coding: utf-8

"""
A module for reading and writing NIfTI-1 files [NIFTI1]_, basically a wrapper for calls
on the Nibabel library [NIFTI2]_.

References
----------
.. [NIFTI1] http://niftilib.sourceforge.net/c_api_html/nifti1_8h-source.html (20180212)
.. [NIFTI2] http://nipy.org/nibabel/ (20180212).
"""

import gzip
import nibabel
import numpy as np
from pathlib import Path

import supervisely.volume.loaders.anatomical_coords as ac
from supervisely.volume.loaders.volume import Volume


def open_image(path, verbose=True, squeeze=False):
    """
    Open a NIfTI-1 image at the given path. The image might have an arbitrary number of dimensions; however, its first
    three axes are assumed to hold its spatial dimensions.

    Parameters
    ----------
    path : str
        The path of the file to be loaded.
    verbose : bool, optional
        If `True` (default), print some meta data of the loaded file to standard output.
    squeeze : bool, optional
        If `True`, remove trailing dimensions of the image volume if they contains a single entry only (default is
        `False`). Note that in this case it has not been tested whether the coordinate transformations from the NIfTI-1
        header still apply.

    Returns
    -------
    Volume
        The resulting 3D image volume, with the ``src_object`` attribute set to the respective
        ``nibabel.nifti1.Nifti1Image`` instance and the desired anatomical world coordinate system ``system`` set to
        "RAS". Relies on the NIfTI header's `get_best_affine()` method to dermine which transformation matrix to use
        (qform or sform).

    Raises
    ------
    IOError
        If something goes wrong.
    """
    # According to the NIfTI-1 specification [1]_, the world coordinate system of NIfTI-1 files is always RAS.
    src_system = "RAS"

    try:
        src_object = nibabel.nifti1.load(path)
    except Exception as e:
        raise IOError(e)

    voxel_data = np.asanyarray(src_object.dataobj)
    if isinstance(voxel_data, np.memmap):
        voxel_data.mode = (
            "c"  # Make sure that no changes happen to data on disk: copy on write
        )
    hdr = src_object.header

    ndim = hdr["dim"][0]
    if ndim < 3:
        raise IOError(
            "Currently only 3D images can be handled. The given image has {} dimension(s).".format(
                ndim
            )
        )

    if verbose:
        print("Loading image:", path)
        print("Meta data:")
        print(hdr)
        print("Image dimensions:", voxel_data.ndim)

    # Squeeze superfluous dimensions (according to the NIfTI-1 specification [1]_, the spatial dimensions are always
    # in front)
    if squeeze:
        voxel_data = __squeeze_dim(voxel_data, verbose)

    mat = hdr.get_best_affine()

    volume = Volume(
        src_voxel_data=voxel_data,
        src_transformation=mat,
        src_system=src_system,
        src_spatial_dimensions=(0, 1, 2),
        system="RAS",
        src_object=src_object,
    )
    return volume


def save_image(path, data, transformation, spatial_dimensions=(0, 1, 2)):
    """
    Save the given image data as a NIfTI image file at the given path.

    Parameters
    ----------
    path : str
        The path for the file to be saved.
    data : array_like
        :math:`N`-dimensional array (:math:`N≥3`) that contains the voxels to be saved. The array is assumed to contain
        a 3D image, i.e. three of its axes (as specified via ``spatial_dimensions``) define its spatial dimensions
        while the remaining :math:`N-3` axes are its time and/or data dimensions.
    transformation : array_like
        :math:`4×4` transformation matrix that maps from ``data``'s voxel indices to a RAS anatomical world coordinate
        system.
    spatial_dimensions : sequence of int, optional
        The three axes that correspond to the spatial dimensions of the ``data`` array (default: (0, 1, 2)). The order
        of the given values is ignored, as the mapping order from voxel indices to the world coordinate system should be
        handled exclusively by the given ``transformation`` matrix.
    """
    data = ac.pull_spatial_dimensions(
        data, spatial_dimensions
    )  # Spatial dimensions must always be in front for NIfTI
    nibabel.Nifti1Image(data, transformation).to_filename(path)


def save_volume(path, volume, src_order=True):
    """
    Save the given ``Volume`` instance as a NIfTI image file at the given path.

    Parameters
    ----------
    path : str
        The path for the file to be saved.
    volume : Volume
        The ``Volume`` instance containing the image data to be saved.
    src_order : bool, optional
        If `True` (default), order the saved voxels as in ``src_data``; if `False`, order the saved voxels as in
        ``aligned_data``. In any case, the correct transformation matrix will be chosen. Furthermore, the three
        spatial dimensions, in accordance with the NIfTI-1 specification [NIFTI1]_, will always end up in the first three
        axes of the saved volume.
    """
    system = "RAS"

    if src_order:
        data = ac.pull_spatial_dimensions(
            volume.src_data, volume.src_spatial_dimensions
        )
        transformation = volume.get_src_transformation(system)
    else:
        data = volume.aligned_data  # Spatial dimensions already in front
        transformation = volume.get_aligned_transformation(system)

    save_image(path, data=data, transformation=transformation)


def __squeeze_dim(data, verbose):
    """
    For arrays with more than three dimensions and the trailing dimensions containing one element only, return a new 3D
    array of the same content. For other arrays, simply return them.

    Parameters
    ----------
    data : array_like
        The array to be repaired
    verbose : bool
        If `True`, print a message in case the dimensions have changed.

    Return
    ------
    array_like
        The result from correction.
    """
    if data.ndim > 3 and np.all(np.asarray(data.shape[3:]) == 1):
        squeezed_data = np.squeeze(data, axis=np.arange(3, data.ndim)).copy()
        if verbose:
            print("{}D array has been corrected to 3D.".format(data.ndim))
    else:
        squeezed_data = data
    return squeezed_data


def compress(path, delete_originals=False):
    """
    Compress the NIfTI file(s) at the given path to `.nii.gz` files. Save the result(s) with the same name(s) in the
    same folder, but with the file extension changed from `.nii` to `.nii.gz`.

    Parameters
    ----------
    path : str
        If a directory path is given, compress all contained .nii files (just in the folder, not in its subfolders). If
        a path to a `.nii` file is given, compress the file.
    delete_originals : bool, option
        If `True`, try to delete the original `.nii` file(s) after compressing (default is `False`).
    """
    path = Path(path).resolve()
    if path.is_dir():
        file_paths = sorted(
            (f.resolve() for f in path.iterdir() if str(f).lower().endswith(".nii")),
            key=lambda p: str(p).lower(),
        )
    else:
        file_paths = [path]

    for in_path in file_paths:
        in_path = str(in_path)
        try:
            print("Compressing {} ...".format(in_path))
            out_path = in_path + ".gz"
            with open(in_path, "rb") as in_file, gzip.open(out_path, "wb") as out_file:
                out_file.writelines(in_file)
            compress_success = True
        except Exception as e:
            print("Compressing {} failed! ({})".format(in_path, e))
            compress_success = False
        if compress_success and delete_originals:
            try:
                print("Deleting {} ...".format(in_path))
                Path(in_path).unlink()
            except Exception as e:
                print("Deleting {} failed! ({})".format(in_path, e))
