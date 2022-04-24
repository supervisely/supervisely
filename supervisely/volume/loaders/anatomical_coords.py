#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform between different 3D anatomical coordinate systems (RAS, LAS etc.)
"""

import numpy as np
from numpy import ma


#: for every uppercase letter defining an anatomical direction, when given as a key, return an uppercase letter that
#: marks the opposite direction in the same anatomical axis.
opposites = {'R': "L", 'A': "P", 'S': "I", 'L': "R", 'P': "A", 'I': "S"}


def permutation_matrix(src, dst):  # TODO: test cases
    """
    Calculate the permutation-reflection matrix that maps axes from the given source anatomical coordinate system to the
    given destination anatomical coordinate system, as well as its inverse.

    Parameters
    ----------
    src : str
        A three-character string that describes the source system (such as "LPS"). Any permutation of {A,P}, {I,S},
        {L,R} (case-insensitive) can be used.
    dst : str
        A three-character string that describes the destination system (such as "RAS").  Any permutation of {A,P},
        {I,S}, {L,R} (case-insensitive) can be used.

    Returns
    -------
    numpy.ndarray
        The :math:`3×3` permutation-reflection matrix that maps coordinates from the ``src`` system to the ``dst``
        system. A minus one signifies a swapped axis direction (e.g. "L" in `src`` becomes "R" in ``dst``). Note that
        the inverse mapping is given by the result's transpose.
    """
    src = src.upper()
    dst = dst.upper()

    # Find the "R/L", "A/P", "S/I" positions
    src_pos = pos(src)
    dst_pos = pos(dst)

    ndim = 3
    dtype = np.int

    # Actually build the transformation matrix
    mat = np.zeros((ndim, ndim), dtype=dtype)
    for i in range(ndim):
        # If the character for the current axis is not the same in the source and destination string, we have to mirror
        # the respective axis (-1), otherwise not (1)
        mat[dst_pos[i], src_pos[i]] = -1 if dst[dst_pos[i]] != src[src_pos[i]] else 1

    return mat


def find_closest_permutation_matrix(trans):
    """
    Find the transformation matrix that *almost* maps voxel axes to original world coordinate axes, but does not
    require interpolation, i.e. the permutation-reflection matrix closest to the given transformation matrix.

    Parameters
    ----------
    trans : array_like
        The :math:`d×d` matrix that represents the original transformations from voxel indices to world coordinates
        (excluding offset).

    Returns
    -------
    numpy.ndarray
        The resulting :math:`d×d` permutation-reflection matrix (containing only integers 0, 1, and -1).
    """
    trans_abs = ma.masked_array(remove_scaling(np.abs(trans)), mask=(np.zeros_like(trans, dtype=np.bool)))

    perm = np.zeros(trans_abs.shape, dtype=np.int)
    # Set the maxima to ±1, keeping track of rows/columns already set to avoid collisions
    while np.sum(~trans_abs.mask) > 0:
        ij_argmax = np.unravel_index(trans_abs.argmax(), trans_abs.shape)
        perm[ij_argmax] = np.sign(trans[ij_argmax])
        trans_abs.mask[ij_argmax[0], :] = True
        trans_abs.mask[:, ij_argmax[1]] = True
    return perm


def remove_scaling(trans):
    """
    Remove scaling from the given :math:`d×d` rotation matrix, i.e. divide its columns by their Euclidean norm.

    Parameters
    ----------
    trans : array_like
        The matrix whose scaling is to be removed; should be given without offset/translational part.

    Returns
    -------
    numpy.ndarray
        New matrix with scaling removed. Columns of zero norm will remain zeros.
    """
    trans = np.asarray(trans)
    scaling = np.linalg.norm(trans, axis=0)
    result = trans * (1 / (scaling + (scaling == 0)))[np.newaxis, :]
    return result


def must_be_flipped(perm):
    """
    Find which axis need to be flipped/reversed in the original array, according to the given permutation-reflection
    matrix.

    Parameters
    ----------
    perm : array_like
        A :math:`d×d` matrix that gives the permutations and reflections for swapping.

    Returns
    -------
    numpy.ndarray
        A :math:`d`-dimensional vector holding a one for the axes of the original array that need to be flipped and a
        zero for the others.
    """
    result = (np.sum(perm, axis=0) < 0).astype(int)
    return result


def offset(perm, shape):
    """
    Calculate the offset to be added on the indices of the original array to end up with the indices of the array that
    results from swapping according to `perm`.

    This means that if `r` is the :math:`4×4` output of this function, then `homogeneous_matrix(perm[:3, :3]) @ r` maps
    form indices in the original array to indices in the swapped array.

    Parameters
    ----------
    perm : array_like
        A :math:`d×d` matrix that gives the permutations and reflections for swapping, where `d` is implied by
        `len(shape)`. If more values are given, the upper left :math:`d×d` area is considered.
    shape : array_like
        Tuple of :math:`d` values that give the shape of the original array (i.e. before swapping).

    Returns
    -------
    numpy.ndarray
        A :math:`(d+1)×(d+1)` matrix that holds the calculated offset as its translational part.
    """
    ndim = len(shape)
    max_indices = np.asarray(shape) - 1
    # Swap if the sign of the respective column's nonzero element is negative -> add offset there
    offset_vector = must_be_flipped(perm[:ndim, :ndim]) * (-max_indices)
    offset_matrix = np.eye(ndim + 1, dtype=np.int)
    offset_matrix[:-1, -1] = offset_vector
    return offset_matrix


def pull_spatial_dimensions(a, spatial_dimensions, sort=True, copy=False):
    """
    Bring the given array's :math:`d` spatial dimensions to the front.

    Parameters
    ----------
    a : array_like
        The :math:`N`-dimensional array (:math:`N≥d`) whose spatial dimensions are to be pulled.
    spatial_dimensions : sequence of int
        A :math:`d`-tuple that gives the positions of the array's spatial dimensions. See ``sort`` for the meaning of
        the values' order.
    sort : bool, optional
        Specify sorting of the spatial dimensions for pulling. If `True` (default), the smallest value of
        ``spatial_dimensions`` will end up as axis 0, the second smallest one as axis 1, etc. If `False`, the first
        value will end up as axis 0, the second one as axis 1, etc.
    copy : bool, optional
        If `False` (default), return a view into the given array `a`; if `True`, return an array that does not share
        data with `a`.

    Returns
    -------
    numpy.ndarray
        An :math:`N`-dimensional array with the spatial dimensions along the first :math:`d` axes. The order of the
        remaining axes is unchanged.
    """
    a = a.copy() if copy else a
    spatial_dimensions = sorted(spatial_dimensions) if sort else spatial_dimensions
    a = np.moveaxis(a, source=spatial_dimensions, destination=np.arange(len(spatial_dimensions)))
    return a


def push_spatial_dimensions(a, spatial_dimensions, sort=True, copy=False):
    """
    Bring the given array's :math:`d` spatial dimensions from the front back to their original position.

    Parameters
    ----------
    a : array_like
        The :math:`N`-dimensional array (:math:`N≥d`) whose spatial dimensions are to be pushed back.
    spatial_dimensions : sequence of int
        A :math:`d`-tuple that gives the original positions of the array's spatial dimensions. See ``sort`` for the
        meaning of the values' order.
    sort : bool, optional
        Specify sorting of the spatial dimensions for pushing. If `True` (default), axis 0 will end up at the smallest
        value of ``spatial_dimensions``, axis 1 will end up at the second smallest value, etc. If `False`, axis 0 will
        end up at the first value, axis 1 will end up at the second value, etc.
    copy : bool, optional
        If `False` (default), return a view into the given array `a`; if `True`, return an array that does not share
        data with `a`.

    Returns
    -------
    numpy.ndarray
        An :math:`N`-dimensional array with the :math:`d` spatial dimensions along their original axes. The order of the
        remaining axes is unchanged.
    """
    a = a.copy() if copy else a
    spatial_dimensions = sorted(spatial_dimensions) if sort else spatial_dimensions
    a = np.moveaxis(a, source=np.arange(len(spatial_dimensions)), destination=spatial_dimensions)
    return a


def swap(a, perm, spatial_dimensions, copy=False):
    """
    Swap the :math:`d` spatial dimensions of the given volume according to the given permutation-reflection matrix.

    Parameters
    ----------
    a : array_like
        The :math:`N`-dimensional array (:math:`N≥d`) whose values are to be swapped.
    perm : array_like
        A :math:`d×d` matrix that gives the permutations and reflections for swapping. If more than :math:`d×d` values
        are given, where :math:`d` is implied by the length of ``spatial_dimensions``, then the upper left :math:`d×d`
        area is considered. The given array should represent a permutation-reflection matrix that maps the coordinate
        axes of one coordinate system exactly onto the axes of another coordinate system.
    spatial_dimensions: sequence of int
        The :math:`d` spatial dimensions of the array. The order of the given values is ignored, as the mapping order
        from one coordinate system to the other should be handled exclusively by the given ``perm`` matrix.
    copy : bool, optional
        If `False` (default), return a view into the given array `a` whenever possible; if `True`, return an array that
        does not share data with `a`.
    Returns
    -------
    numpy.ndarray
        The :math:`N`-dimensional array that results from swapping. All axes except for the swapped spatial dimensions
        remain in their original positions.
    """
    a = a.copy() if copy else a

    ndim = len(spatial_dimensions)
    perm = perm[:ndim, :ndim]
    validate_permutation_matrix(perm)

    # Bring spatial dimensions to the front, i.e. to the first ndim axes
    a = pull_spatial_dimensions(a, spatial_dimensions)

    # Reverse values in the necessary axes of the first ndim dimensions
    flips = must_be_flipped(perm[:ndim, :ndim])
    a = a[tuple(slice(None, None, -1 if f else None) for f in flips)]

    # Permute first ndim axes
    permutations = np.arange(a.ndim)
    permutations[:ndim] = (np.abs(perm) @ np.arange(ndim)).astype(np.int)
    a = np.transpose(a, axes=permutations)

    # Push ndim spatial dimensions back to their original axes and return
    a = push_spatial_dimensions(a, spatial_dimensions)
    return a


def pos(system):
    """
    Return a tuple `(rl, ap, si)` where `rl` holds the index of "R" or "L", `ap` holds the index of "A" or "P",
    and `si` holds the index of "S" or "I" in the given string.

    Parameters
    ----------
    system : str
        String to be processed. Any permutation of {A,P}, {I,S}, {L,R} (case-insensitive) can be used.

    Returns
    -------
    tuple
        The resulting character positions (0, 1, or 2).
    """
    return index(system, "R"), index(system, "A"), index(system, "S")


def index(system, character):
    """
    Get the index that the given character or its anatomical opposite ("A" vs. "P", "I" vs. "S", "L" vs. "R") has in
    the given string.

    Parameters
    ----------
    system : str
        String to be processed. Any permutation of {A,P}, {I,S}, {L,R} (case-insensitive) can be used.
    character : str
        Character to be found. One of "A", "P", "I", "S", "L", "R" (case insensitive).

    Returns
    -------
    int
        Index of the given character.

    Raises
    ------
    ValueError
        If the given character or its anatomical opposite cannot be found.
    """
    system = system.upper()
    character = character.upper()

    i = system.find(character)
    i = system.index(opposites[character]) if i == -1 else i
    # ^ str.find() returns -1 for mismatch, while str.index() raises an error

    return i


def homogeneous_vector(v):
    """
    Make the given vector(s) homogeneous: for a :math:`d`-element vector, append one and make it a :math:`d+1`-element
    vector; for a :math:`d×n` array of vectors, append a row of ones and make it a :math:`(d+1)×n` array.

    Parameters
    ----------
    v : array_like
        Either a :math:`d`-element vector or a :math:`d×n` array of vectors.

    Returns
    -------
    numpy.ndarray
        A :math:`d+1`-element vector or :math:`(d+1)×n` array.
    """
    v = np.asarray(v)
    if v.ndim == 1:
        v_h = np.r_[v, 1]
    elif v.ndim == 2:
        v_h = np.r_[v, np.ones(v.shape[-1], dtype=v.dtype)[np.newaxis, :]]
    else:
        raise ValueError("Cannot handle array of shape {}!".format(v.shape))
    return v_h


def homogeneous_matrix(m):
    """
    Make the given :math:`d×d` matrix homogeneous: place and return it in the top left corner of a :math:`(d+1)×(d+1)`
    identity matrix.

    Parameters
    ----------
    m : array_like
        The :math:`d×d` matrix to be handled.

    Returns
    -------
    numpy.ndarray
        The resulting :math:`(d+1)×(d+1)` matrix.
    """
    m = np.asarray(m)
    assert m.shape[0] == m.shape[1], "Cannot handle array of shape {}!".format(m.shape)
    m_h = np.eye(m.shape[0] + 1, dtype=m.dtype)
    m_h[:-1, :-1] = m
    return m_h


def get_rotational_part(trans):
    """
    Get the :math:`d×d` rotational part of a :math:`(d+1)×(d+1)` transformation matrix.

    Parameters
    ----------
    trans : array_like
        The given transformation matrix.

    Returns
    -------
    numpy.ndarray
        The rotational part, with potential scaling removed.
    """
    trans = np.asarray(trans)
    assert trans.shape[0] == trans.shape[1]
    ndim = trans.shape[0] - 1

    result = remove_scaling(trans[:ndim, :ndim])
    return result


def transformation_for_new_coordinate_system(trans, sold2snew):
    """
    Calculate a new transformation matrix, based on a given transformation matrix (which maps from given voxel indices
    to a given coordinate system) and a permutation-reflection matrix (which maps from the given coordinate system to
    the desired coordinate system).

    Parameters
    ----------
    trans: array_like
        The given :math:`(d+1)×(d+1)` transformation matrix.
    sold2snew: array_like
        The :math:`d×d` permutation-reflection matrix.

    Returns
    -------
    numpy.ndarray
        The resulting new :math:`(d+1)×(d+1)` transformation matrix.
    """
    vold2cold = trans
    result = homogeneous_matrix(sold2snew) @ vold2cold
    return result


def transformation_for_new_voxel_alignment(trans, vnew2vold):
    """
    Calculate a new transformation matrix, based on a given transformation matrix (which maps from given voxel indices
    to a given coordinate system) and a voxel-alignment transformation matrix (which maps from the *desired* voxel
    indices to the *given* voxel indices, including offsets).

    Parameters
    ----------
    trans: array_like
        The given :math:`(d+1)×(d+1)` transformation matrix.
    vnew2vold: array_like
        The :math:`(d+1)×(d+1)` voxel-alignment matrix, including offset.

    Returns
    -------
    numpy.ndarray
        The resulting new :math:`(d+1)×(d+1)` transformation matrix.
    """
    vold2cold = np.asarray(trans)  # Ensure that "@" will work
    result = vold2cold @ vnew2vold
    return result


def validate_permutation_matrix(perm):
    """
    Validate a permutation-reflection matrix. A matrix is considered valid if (1) its determinant is either exactly 1 or
    or exactly -1 and (2) all of its values are either -1, 0, or 1.

    Parameters
    ----------
    perm : array_like
        The :math:`d×d` matrix to be validated.

    Returns
    -------
    None
        Simply return if the matrix is valid.

    Raises
    ------
    ValueError
        If the matrix is invalid.
    """
    msg = ""
    if np.abs(np.linalg.det(perm)) != 1:
        msg = "the matrix determinant is neither -1 nor 1"
    elif not np.all(np.isin(perm, [-1, 0, 1])):
        msg = "at least one matrix element is not in {-1, 0, 1}"
    if msg:
        raise ValueError("The given matrix is not valid: {}.".format(msg))


def validate_transformation_matrix(mat, tol=1e-1):
    """
    Validate a transformation matrix. A :math:`d×d` matrix is considered valid if (1) its :math:`(d-1)×(d-1)` rotational
    part has a determinant of absolute value close to one, (2) its last row consists of zeros with a trailing one.

    Parameters
    ----------
    mat : array_like
        The :math:`d×d` matrix to be validated.
    tol : float
        Tolerance for absolute value `v` of the rotational part's determinant: if :math:`(1 - tol) <= v <= (1 + tol)`,
        then `v` is considered close to one (default: 1e-2; arbitrary choice).

    Returns
    -------
    None
        Simply return if the matrix is valid.

    Raises
    ------
    ValueError
        If the matrix is invalid.
    """
    # Account for potential scaling
    rot_part = get_rotational_part(mat)

    abs_det = np.abs(np.linalg.det(rot_part))
    msg = ""
    if not ((1 - tol) <= abs_det <= (1 + tol)):
        msg = "the determinant's absolute value {} is not close to one".format(abs_det)
    elif np.any(mat[-1, :-1] != 0):
        msg = "the last row contains non-zero values"
    elif mat[-1, -1] != 1:
        msg = "the bottom right value is not one, but {}".format(mat[-1, -1])
    if msg:
        raise ValueError("the given matrix is not valid: {}.".format(msg))
