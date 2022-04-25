#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Provide a class that represents 3D scan volumes in a desired anatomical coordinate system.
"""

import numpy as np

import supervisely.volume.loaders.anatomical_coords as ac


class Volume:
    """
    Volume(src_voxel_data, src_transformation, src_system,
           src_spatial_dimensions=(0, 1, 2), system="RAS", src_object=None)

    Return an object that represents 3D image volumes in a desired anatomical world coordinate system (``system``;
    default is "RAS"), based on (1) an array that holds the voxels (``src_voxel_data``) and (2) a transformation matrix
    (``src_transformation``) that holds the mapping from voxel indices to (3) some potentially different given
    anatomical world coordinate system (``src_system``). The class is meant to serve as a layer on top of specific
    image formats (with different coordinate system conventions).

    It is also meant to make dealing with the voxel data a little simpler: when accessing voxels via the field
    ``aligned_data``, the voxel data axes are aligned with the anatomical world coordinate system axes as closely as
    is possible without reinterpolating the image.

    Parameters
    ----------
    src_voxel_data : array_like
        An :math:`N`-dimensional array (:math:`N≥3`) that contains the image voxels, arranged to match the coordinate
        transformation matrix ``src_transformation``. The array is assumed to contain a 3D image, i.e. three of its
        axes (as specified via ``src_spatial_dimensions``) define its spatial dimensions while the remaining
        :math:`N-3` axes are its time and/or data dimensions. If :math:`N=3`, scalar data is assumed.
    src_transformation : array_like
        A :math:`4×4` matrix that describes the mapping from voxel indices in ``src_voxel_data`` to the given anatomical
        world coordinate system ``src_system``.
    src_system : str
        A three-character string that describes the anatomical world coordinate system for the provided
        ``src_transformation`` matrix. Any permutation of {A,P}, {I,S}, {L,R} (case-insensitive) can be used. For
        example, for voxels and a transformation matrix provided by a DICOM loading library, this should usually be
        "LPS", as this is the assumed world coordinate system of the DICOM standard.
    src_spatial_dimensions : sequence of int, optional
        The three axes that correspond to the spatial dimensions of the ``src_voxel_data`` array (default: (0, 1, 2)).
        The order of the given values is ignored, as the mapping order from voxel indices to the world coordinate system
        should be handled exclusively by the given ``src_transformation`` matrix.
    system : str, optional
        A three-character string similar to ``src_system``. However, ``system`` should describe the anatomical world
        coordinate system that the *user* assumes/desires. It will also determine the arrangement of the voxel data for
        the ``aligned_data`` representation (default: "RAS").
    src_object : object, optional
        The original object that was created by the image loading library (nibabel, pydicom, ...) to get the provided
        ``src_voxel_data`` and ``src_transformation`` -- for debugging, for example (default: None).
    """

    def __init__(
        self,
        src_voxel_data,
        src_transformation,
        src_system,
        src_spatial_dimensions=(0, 1, 2),
        system="RAS",
        src_object=None,
    ):

        self.__src_system = src_system
        self.__user_system = None

        # Mapping from ``src_data``'s voxel indices to the source anatomical coordinate system
        ac.validate_transformation_matrix(src_transformation)
        self.__vsrc2csrc_4x4 = src_transformation

        self.__src_object = src_object
        self.__src_spacing = None  # Voxel spacing for ``src_data``
        self.__src_data = src_voxel_data  # The source voxel data
        self.__vsrc2cuser_4x4 = None
        # ^ Mapping from ``src_data``'s voxel indices to the desired anatomical coordinate system
        self.__src_spatial_dimensions = tuple(
            sorted(src_spatial_dimensions)
        )  # Remaining dimensions are time or data
        self.__src_spatial_shape = tuple(
            self.__src_data.shape[i] for i in self.__src_spatial_dimensions
        )

        self.__aligned_spacing = None
        self.__aligned_data = None
        self.__vuser2cuser_4x4 = None
        # ^ Mapping from ``aligned_data``'s voxel indices to the desired anatomical coordinate system

        # Mapping from ``src_data``'s voxel indices to ``aligned_data`` voxel indices and vice versa (including
        # offset into the array)
        self.__vsrc2vuser_4x4 = None
        self.__vuser2vsrc_4x4 = None

        self.system = system  # Initialize the remaining empty fields

    @property
    def system(self):
        """
        str:
            The desired anatomical world coordinate system as a three-character string. Any permutation of {A,P}, {I,S},
            {L,R} (case-insensitive) can be used. When being set, fields like ``aligned_data``, ``aligned_spacing``,
            ``aligned_transformation``, and ``src_to_aligned_transformation`` will be adjusted accordingly.
        """
        return self.__user_system

    @system.setter
    def system(self, value):

        new_system = value.upper()
        if new_system != self.__user_system:
            self.__on_system_change(new_system)

    def __on_system_change(self, new_system):

        ndim = 3
        self.__user_system = new_system

        # Transform: given source array indices -> source system coordinates (known)
        vsrc2csrc_4x4 = self.__vsrc2csrc_4x4

        # Swap: given source array axes -> source system axes
        vsrc2ssrc_3x3 = ac.find_closest_permutation_matrix(vsrc2csrc_4x4[:ndim, :ndim])
        # Swap: source system axes -> user system axes
        ssrc2suser_3x3 = ac.permutation_matrix(self.__src_system, new_system)
        # Swap: given source array axes -> user system axes
        vsrc2suser_3x3 = ssrc2suser_3x3 @ vsrc2ssrc_3x3

        offset_4x4 = ac.offset(vsrc2suser_3x3, self.__src_spatial_shape)
        # Transform: given source array indices -> user system aligned array indices
        vsrc2vuser_4x4 = ac.homogeneous_matrix(vsrc2suser_3x3) @ offset_4x4
        # Transform: user system aligned array indices -> given source array indices
        vuser2vsrc_4x4 = np.round(np.linalg.inv(vsrc2vuser_4x4)).astype(
            vsrc2vuser_4x4.dtype
        )

        # Transform: given source array indices -> user system coordinates
        vsrc2cuser_4x4 = ac.transformation_for_new_coordinate_system(
            trans=vsrc2csrc_4x4, sold2snew=ssrc2suser_3x3
        )
        # Transform: user system aligned array indices -> user system coordinates
        vuser2cuser_4x4 = ac.transformation_for_new_voxel_alignment(
            trans=vsrc2cuser_4x4, vnew2vold=vuser2vsrc_4x4
        )

        self.__vsrc2vuser_4x4 = vsrc2vuser_4x4
        self.__vuser2vsrc_4x4 = vuser2vsrc_4x4
        self.__vsrc2cuser_4x4 = vsrc2cuser_4x4
        self.__vuser2cuser_4x4 = vuser2cuser_4x4

        # Recalculate voxel sizes ("spacing")
        self.__src_spacing = tuple(np.linalg.norm(vsrc2csrc_4x4[:ndim, :ndim], axis=0))
        self.__aligned_spacing = tuple(
            np.linalg.norm(vuser2cuser_4x4[:ndim, :ndim], axis=0)
        )

        # Actually swap the given source array, then bring the spatial dimensions to the front
        aligned_volume = ac.swap(
            self.__src_data, vsrc2vuser_4x4, self.__src_spatial_dimensions
        )
        aligned_volume = ac.pull_spatial_dimensions(
            aligned_volume, self.__src_spatial_dimensions
        )
        self.__aligned_data = aligned_volume

    @property
    def src_system(self):
        """
        str:
            The original anatomical world coordinate system as a three-character string.
        """
        return self.__src_system

    @property
    def src_spatial_dimensions(self):
        """
        tuple:
            In ascending order , the three axes that contain spatial dimensions in ``src_data``.
        """
        return self.__src_spatial_dimensions

    @property
    def src_object(self):
        """
        object:
            The object that originally was returned by the image loading library (or None).
        """
        return self.__src_object

    @property
    def src_transformation(self):
        """
        numpy.ndarray:
            The :math:`4×4` transformation matrix that maps from ``src_data``'s voxel indices to the *original*
            anatomical world coordinate system ``src_system`` (new copy).
        """
        return self.__vsrc2csrc_4x4.copy()

    @property
    def aligned_transformation(self):
        """
        numpy.ndarray:
            The :math:`4×4` transformation matrix that maps from ``aligned_data``'s voxel indices to the *desired*
            anatomical world coordinate system ``system`` (new copy).
        """
        return self.__vuser2cuser_4x4.copy()

    @property
    def src_to_aligned_transformation(self):
        """
        numpy.ndarray:
            The :math:`4×4` transformation matrix that maps from ``src_data``'s voxel indices to the *desired*
            anatomical world coordinate system ``system`` (new copy).
        """
        return self.__vsrc2cuser_4x4.copy()

    @property
    def src_data(self):
        """
        numpy.ndarray:
            The :math:`N`-dimensional Numpy array (:math:`N≥3`) that contains the original voxel data.
        """
        return self.__src_data

    @property
    def src_volume(self):
        """
        numpy.ndarray:
            Alias for ``src_data`` to maintain backward compatibility.
        """
        return self.__src_data

    @property
    def aligned_data(self):
        """
        numpy.ndarray:
            The :math:`N`-dimensional Numpy array (:math:`N≥3`) that contains the image information with the voxel data
            axes aligned to the desired anatomical world coordinate system ``system`` as closely as is possible without
            reinterpolation. The three spatial dimensions are brought to the front (i.e. axes 0, 1, 2), the remaining
            dimensions (time and/or data dimensions) are brought to the back, keeping their original order. This means,
            for example, if ``system`` is "RAS", then ``aligned_data`` will hold an array where increasing the index
            on axis 0 will reach a voxel coordinate that is typically more to the right side of the imaged subject,
            increasing the index on axis 1 will reach a voxel coordinate that is more anterior, and increasing the index
            on axis 2 will reach a voxel coordinate that is more superior.
        """
        return self.__aligned_data

    @property
    def aligned_volume(self):
        """
        numpy.ndarray:
            Alias for ``aligned_data`` to maintain backward compatibility.
        """
        return self.__aligned_data

    @property
    def src_spacing(self):
        """
        tuple:
            The spacing of ``src_data`` as a three-tuple in world coordinate system units per voxel.
        """
        return self.__src_spacing

    @property
    def aligned_spacing(self):
        """
        tuple:
            The spacing of ``aligned_data`` as a three-tuple in world coordinate system units per voxel.
        """
        return self.__aligned_spacing

    def get_src_transformation(self, system):
        """
        Get a transformation matrix that maps from ``src_data``'s voxel indices to the given anatomical world
        coordinate system.

        Parameters
        ----------
        system : str
            A three-character string that describes the anatomical world coordinate system. Any permutation of {A,P},
            {I,S}, {L,R} (case-insensitive) can be used.

        Returns
        -------
        numpy.ndarray
            The resulting :math:`4×4` transformation matrix.

        See also
        --------
        get_aligned_transformation : Same transformation, but for ``aligned_data``.
        """
        sold2snew_3x3 = ac.permutation_matrix(self.__src_system, system)
        return ac.transformation_for_new_coordinate_system(
            trans=self.__vsrc2csrc_4x4, sold2snew=sold2snew_3x3
        )

    def get_aligned_transformation(self, system):
        """
        Get a transformation matrix that maps from ``aligned_data``'s voxel indices to the given anatomical world
        coordinate system.

        Parameters
        ----------
        system : str
            A three-character string that describes the anatomical world coordinate system. Any permutation of {A,P},
            {I,S}, {L,R} (case-insensitive) can be used.

        Returns
        -------
        numpy.ndarray
            The resulting :math:`4×4` transformation matrix.

        See also
        --------
        get_src_transformation : Same transformation, but for ``src_data``.
        """
        vsrc2csys_4x4 = self.get_src_transformation(system=system)
        return ac.transformation_for_new_voxel_alignment(
            trans=vsrc2csys_4x4, vnew2vold=self.__vuser2vsrc_4x4
        )

    def copy(self, deep=True):
        """
        Create a shallow(er) or deep(er) copy of the current instance.

        Parameters
        ----------
        deep : bool, optional
            If `True` (default), a copy of the ``src_data`` Numpy array will be created for the new instance; if
            `False`, the array will be shared by both instances. In either case, (1) ``src_object`` will be shared by
            both instances and (2) the transformation matrices will be copies for the new instance.

        Returns
        -------
        Volume
            A copy of the current instance.
        """
        src_voxel_data = self.__src_data.copy() if deep else self.__src_data
        return Volume(
            src_voxel_data=src_voxel_data,
            src_transformation=self.__vsrc2csrc_4x4.copy(),
            src_system=self.__src_system,
            system=self.__user_system,
            src_object=self.__src_object,
        )

    def copy_like(self, template, src_spatial_dimensions=None, deep=True):
        """
        Create a copy of the current instance, rearranging the following data to match the respective entries of
        ``template``: (1) ``src_data``, (2) ``src_system``, (3) ``aligned_data``, (4) ``system``.

        Parameters
        ----------
        template : Volume
            The instance whose order of ``src_data`` voxels and whose world coordinate systems should be adopted.
        src_spatial_dimensions : None or sequence of int, optional
            The axes on which the spatial dimensions should end up in the new instance's ``src_data``. If None
            (default), the positions of the spatial axes in the ``src_data`` of the ``template`` will be used (this
            implies that the current instance has at least as many axes as the ``template``). Otherwise, the three given
            values will define the positions. The order of the given values is ignored, as the mapping order from one
            coordinate system to the other should be handled exclusively by the transformation matrices and coordinate
            system definitions.
        deep : bool, optional
            If `True` (default), a copy of the current instance's ``aligned_data`` Numpy array will be created for the
            new instance; if `False`, the ``*_volume`` arrays of the new instance will be a view into said array
            whenever possible. In either case, (1) ``src_object`` will be shared by both instances and (2) the
            transformation matrices will be copies for the new instance.

        Returns
        -------
        Volume
            A rearranged copy of the current instance.
        """
        current_instance = self
        tpl_ssrc = template.__src_system

        # Get mapping from template's user-aligned voxel indices to its source voxel indices (ignoring offsets)
        tpl_vuser_2_tpl_vsrc_3x3 = template.__vuser2vsrc_4x4[:3, :3]
        # Get mapping from current instance's user system to template's user system
        tpl_suser = template.__user_system
        cur_suser = current_instance.__user_system
        cur_suser_2_tpl_suser_3x3 = ac.permutation_matrix(src=cur_suser, dst=tpl_suser)

        # Combine, calculate necessary offset, and actually swap current instance's aligned array respectively
        cur_suser_2_tpl_vsrc_3x3 = tpl_vuser_2_tpl_vsrc_3x3 @ cur_suser_2_tpl_suser_3x3
        offset_4x4 = ac.offset(
            cur_suser_2_tpl_vsrc_3x3, current_instance.__aligned_data.shape[:3]
        )
        cur_vuser_2_tpl_vsrc_4x4 = (
            ac.homogeneous_matrix(cur_suser_2_tpl_vsrc_3x3) @ offset_4x4
        )
        cur_aligned_volume_swapped = ac.swap(
            current_instance.__aligned_data,
            cur_vuser_2_tpl_vsrc_4x4,
            spatial_dimensions=(0, 1, 2),
            copy=deep,
        )

        # Move spatial dimensions for the current instances to the defined positions
        src_spatial_dimensions = (
            template.__src_spatial_dimensions
            if src_spatial_dimensions is None
            else src_spatial_dimensions
        )
        ac.push_spatial_dimensions(cur_aligned_volume_swapped, src_spatial_dimensions)

        # Calculate respective transformation to world coordinates for the swapped aligned array
        swapped_vsrc_2_cur_cuser_4x4 = ac.transformation_for_new_voxel_alignment(
            current_instance.__vuser2cuser_4x4, np.linalg.inv(cur_vuser_2_tpl_vsrc_4x4)
        )
        swapped_vsrc_2_tpl_csrc_4x4 = ac.transformation_for_new_coordinate_system(
            swapped_vsrc_2_cur_cuser_4x4,
            ac.permutation_matrix(src=cur_suser, dst=tpl_ssrc),
        )

        return Volume(
            src_voxel_data=cur_aligned_volume_swapped,
            src_transformation=swapped_vsrc_2_tpl_csrc_4x4,
            src_system=tpl_ssrc,
            src_spatial_dimensions=src_spatial_dimensions,
            system=tpl_suser,
            src_object=current_instance.__src_object,
        )

    # TODO: Add a print method for nice output
