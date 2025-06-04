# coding: utf-8
"""Functions for processing volumes"""


import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pydicom
import SimpleITK as sitk
import stringcase
from trimesh import Trimesh

import supervisely.volume.nrrd_encoder as nrrd_encoder
from supervisely import logger
from supervisely.geometry.mask_3d import Mask3D
from supervisely.io.fs import get_file_ext, get_file_name, list_files_recursively
from supervisely.volume.stl_converter import matrix_from_nrrd_header

# Do NOT use directly for extension validation. Use is_valid_ext() /  has_valid_ext() below instead.
ALLOWED_VOLUME_EXTENSIONS = [".nrrd", ".dcm"]


class UnsupportedVolumeFormat(Exception):
    pass


def get_extension(path: str):
    """
    Get extension for given path.

    :param path: Path to volume.
    :type path: str
    :return: Path extension
    :rtype: str
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "src/upload/folder/CTACardio.nrrd"
        ext = sly.volume.get_extension(path=path) # .nrrd
    """

    # magic.from_file("path", mime=True)
    # for nrrd:
    # application/octet-stream
    # for nifti(nii):
    # application/octet-stream
    # for dicom:
    # "application/dicom"

    ext = get_file_ext(path)
    if ext in ALLOWED_VOLUME_EXTENSIONS:
        return ext

    # case when dicom file does not have an extension
    import magic

    mime = magic.from_file(path, mime=True)
    if mime == "application/dicom":
        return ".dcm"
    return None


def is_valid_ext(ext: str) -> bool:
    """
    Checks if given extension is supported.

    :param ext: Volume file extension.
    :type ext: str
    :return: True if extensions is in the list of supported extensions else False
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        sly.volume.is_valid_ext(".nrrd")  # True
        sly.volume.is_valid_ext(".mp4") # False
    """

    if type(ext) is not str:
        return False

    return ext.lower() in ALLOWED_VOLUME_EXTENSIONS


def has_valid_ext(path: str) -> bool:
    """
    Checks if Volume file from given path has supported extension.

    :param path: Path to volume file.
    :type path: str
    :return: True if Volume file has supported extension else False
    :rtype: :class:`bool`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        volume_path = "/home/admin/work/volumes/vol_01.nrrd"
        sly.volume.has_valid_ext(volume_path) # True
    """

    return is_valid_ext(get_extension(path))


def validate_format(path: str):
    """
    Raise error if Volume file from given path couldn't be read or file extension is not supported.

    :param path: Path to Volume file.
    :type path: str
    :raises: :class:`UnsupportedVolumeFormat` if Volume file from given path couldn't be read or file extension is not supported.
    :return: None
    :rtype: :class:`NoneType`
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        volume_path = "/home/admin/work/volumes/vol_01.mp4"
        sly.volume.validate_format(volume_path)
        # File /home/admin/work/volumes/vol_01.mp4 has unsupported volume extension. Supported extensions: [".nrrd", ".dcm"].
    """

    if not has_valid_ext(path):
        raise UnsupportedVolumeFormat(
            f"File {path} has unsupported volume extension. Supported extensions: {ALLOWED_VOLUME_EXTENSIONS}"
        )


def is_valid_format(path: str) -> bool:
    """
    Checks if a given file has a supported format.
    :param path: Path to file.
    :type path: str
    :return: True if file format in list of supported Volume formats, False - in otherwise
    :rtype: :class:`bool`
    :Usage example:
         .. code-block:: python
            import supervisely as sly
            sly.volume.is_valid_format('/volumes/dcm01.dcm') # True
            sly.volume.is_valid_format('/volumes/nrrd.py') # False
    """

    try:
        validate_format(path)
        return True
    except UnsupportedVolumeFormat:
        return False


def rescale_slope_intercept(value: float, slope: float, intercept: float) -> float:
    """
    Rescale intensity value using the given slope and intercept.

    :param value: The intensity value to be rescaled.
    :type value: float
    :param slope: The slope for rescaling.
    :type slope: float
    :param intercept: The intercept for rescaling.
    :type intercept: float
    :return: The rescaled intensity value.
    :rtype: float

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        meta["intensity"]["min"] = sly.volume.volume.rescale_slope_intercept(
            meta["intensity"]["min"],
            meta["rescaleSlope"],
            meta["rescaleIntercept"],
        )
    """

    return value * slope + intercept


def normalize_volume_meta(meta: dict) -> dict:
    """
    Normalize volume metadata.

    :param meta: Metadata of the volume.
    :type meta: dict
    :return: Normalized volume metadata.
    :rtype: dict

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        normalized_meta = sly.volume.volume.volume.normalize_volume_meta(volume_meta)

        print(normalized_meta)
        # Output:
        # {
        #     'ACS': 'RAS',
        #     'channelsCount': 1,
        #     'dimensionsIJK': {'x': 512, 'y': 512, 'z': 139},
        #     'directions': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        #     'intensity': {'max': 3071.0, 'min': -3024.0},
        #     'origin': (-194.238403081894, -217.5384061336518, -347.7500000000001),
        #     'rescaleIntercept': 0,
        #     'rescaleSlope': 1,
        #     'spacing': (0.7617189884185793, 0.7617189884185793, 2.5),
        #     'windowCenter': 23.5,
        #     'windowWidth': 6095.0
        # }
    """

    meta["intensity"]["min"] = rescale_slope_intercept(
        meta["intensity"]["min"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    meta["intensity"]["max"] = rescale_slope_intercept(
        meta["intensity"]["max"],
        meta["rescaleSlope"],
        meta["rescaleIntercept"],
    )

    if "windowWidth" not in meta:
        meta["windowWidth"] = meta["intensity"]["max"] - meta["intensity"]["min"]

    if "windowCenter" not in meta:
        meta["windowCenter"] = meta["intensity"]["min"] + meta["windowWidth"] / 2

    return meta


def read_dicom_serie_volume_np(paths: List[str], anonymize=True) -> Tuple[np.ndarray, dict]:
    """
    Read DICOM series volumes with given paths.

    :param paths: Paths to DICOM volume files.
    :type paths: List[str]
    :param anonymize: Specify whether to hide PatientID and PatientName fields.
    :type anonymize: bool
    :return: Volume data in NumPy array format and dictionary with metadata
    :rtype: Tuple[np.ndarray, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        volume_path = ["/home/admin/work/volumes/vol_01.nrrd"]
        volume_np, meta = sly.volume.read_dicom_serie_volume_np(volume_path)
    """

    import SimpleITK as sitk

    sitk_volume, meta = read_dicom_serie_volume(paths, anonymize=anonymize)
    # for debug:
    # sitk.WriteImage(sitk_volume, "/work/output/sitk.nrrd", useCompression=False, compressionLevel=9)
    # with open("/work/output/test.nrrd", "wb") as file:
    #     file.write(b)
    volume_np = sitk.GetArrayFromImage(sitk_volume)
    volume_np = np.transpose(volume_np, (2, 1, 0))
    return volume_np, meta


_anonymize_tags = ["PatientID", "PatientName"]
_default_dicom_tags = [
    "SeriesInstanceUID",
    "Modality",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
    "PhotometricInterpretation",
]
_default_dicom_tags.extend(_anonymize_tags)

_photometricInterpretationRGB = set(
    [
        "RGB",
        "PALETTE COLOR",
        "YBR_FULL",
        "YBR_FULL_422",
        "YBR_PARTIAL_422",
        "YBR_PARTIAL_420",
        "YBR_RCT",
    ]
)


def read_dicom_tags(
    path: str,
    allowed_keys: Union[None, List[str]] = _default_dicom_tags,
    anonymize: bool = True,
):
    """
    Read DICOM tags from a DICOM file.

    :param path: Path to the DICOM file.
    :type path: str
    :param allowed_keys: List of allowed DICOM keywords to be extracted. Default is None, which means all keywords are allowed.
    :type allowed_keys: Union[None, List[str]], optional
    :param anonymize: Flag to indicate whether to anonymize certain tags or not.
    :type anonymize: bool, optional
    :return: Dictionary containing the extracted DICOM tags.
    :rtype: dict

    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "src/upload/Dicom_files/nnn.dcm"
        dicom_tags = sly.volume.read_dicom_tags(path=path)
    """

    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    vol_info = {}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        tag = pydicom.tag.Tag(k.split("|")[0], k.split("|")[1])
        keyword = pydicom.datadict.keyword_for_tag(tag)
        if allowed_keys is not None and keyword not in allowed_keys:
            continue
        if anonymize is True and keyword in _anonymize_tags:
            v = "anonymized"
        keyword = stringcase.camelcase(keyword)
        vol_info[keyword] = v
        if keyword in [
            "windowCenter",
            "windowWidth",
            "rescaleIntercept",
            "rescaleSlope",
        ]:
            vol_info[keyword] = float(vol_info[keyword].split("\\")[0])
        elif keyword == "photometricInterpretation" and v in _photometricInterpretationRGB:
            vol_info["channelsCount"] = 3
    return vol_info


def encode(volume_np: np.ndarray, volume_meta: dict) -> bytes:
    """
    Encodes a volume from NumPy format into a NRRD format.

    :param volume_np: NumPy array representing the volume data.
    :type volume_np: np.ndarray
    :param volume_meta: Metadata of the volume.
    :type volume_meta: dict

    :return: Encoded volume data in bytes.
    :rtype: bytes

    :Usage example:

     .. code-block:: python

        import numpy as np
        import supervisely as sly

        volume_np = np.random.rand(256, 256, 256)
        volume_meta = {
            "ACS": "RAS",
            "channelsCount": 1,
            "dimensionsIJK": { "x": 512, "y": 512, "z": 139 },
            "directions": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "intensity": { "max": 3071.0, "min": -3024.0 },
            "origin": (-194.238403081894, -217.5384061336518, -347.7500000000001),
            "rescaleIntercept": 0,
            "rescaleSlope": 1,
            "spacing": (0.7617189884185793, 0.7617189884185793, 2.5),
            "windowCenter": 23.5,
            "windowWidth": 6095.0
        }

        encoded_volume = sly.volume.encode(volume_np, volume_meta)
    """

    directions = np.array(volume_meta["directions"]).reshape(3, 3)
    directions *= volume_meta["spacing"]

    volume_bytes = nrrd_encoder.encode(
        volume_np,
        header={
            "encoding": "gzip",
            # "space": "left-posterior-superior",
            "space": "right-anterior-superior",
            "space directions": directions.T.tolist(),
            "space origin": volume_meta["origin"],
        },
        compression_level=1,
    )

    # with open("/work/output/test.nrrd", "wb") as file:
    #     file.write(volume_bytes)

    return volume_bytes


def inspect_dicom_series(root_dir: str, logging: bool = True) -> dict:
    """
    Search for DICOM series in the directory and its subdirectories.
    If several series with the same UID are found in the directory, then the series are numbered in the format: "series_uid_01", "series_uid_02", etc.

    :param root_dir: Directory path with volumes.
    :type root_dir: str
    :param logging: Specify whether to print logging messages.
    :type logging: bool
    :return: Dictionary with DICOM volumes IDs and corresponding file names.
    :rtype: dict
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "src/upload/Dicom_files/"
        series_infos = sly.volume.inspect_dicom_series(root_dir=path)
    """
    import SimpleITK as sitk

    found_series = {}
    for d in os.walk(root_dir):
        dir = d[0]
        reader = sitk.ImageSeriesReader()
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        series_found = reader.GetGDCMSeriesIDs(dir)
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
        if logging:
            logger.info(f"Found {len(series_found)} series in directory {dir}")
        for serie in series_found:
            dicom_names = reader.GetGDCMSeriesFileNames(dir, serie)
            new_key = serie
            new_suffix = 1
            while new_key in found_series:
                new_key = "{}_{:02d}".format(serie, new_suffix)
                new_suffix += 1
            found_series[new_key] = dicom_names
    if logging:
        logger.info(f"Total {len(found_series)} series in directory {root_dir}")
    return found_series


def _sitk_image_orient_ras(sitk_volume: sitk.Image) -> sitk.Image:
    import SimpleITK as sitk

    if sitk_volume.GetDimension() == 4 and sitk_volume.GetSize()[3] == 1:
        sitk_volume = sitk_volume[:, :, :, 0]

    sitk_volume = sitk.DICOMOrient(sitk_volume, "RAS")
    # RAS reorient image using filter
    # orientation_filter = sitk.DICOMOrientImageFilter()
    # orientation_filter.SetDesiredCoordinateOrientation("RAS")
    # sitk_volume = orientation_filter.Execute(sitk_volume)

    # https://discourse.itk.org/t/getdirection-and-getorigin-for-simpleitk-c-implementation/3472/8
    origin = list(sitk_volume.GetOrigin())
    directions = list(sitk_volume.GetDirection())
    origin[0] *= -1
    origin[1] *= -1
    directions[0] *= -1
    directions[1] *= -1
    directions[3] *= -1
    directions[4] *= -1
    directions[6] *= -1
    directions[7] *= -1
    sitk_volume.SetOrigin(origin)
    sitk_volume.SetDirection(directions)
    return sitk_volume


def read_dicom_serie_volume(paths: List[str], anonymize: bool = True) -> Tuple[sitk.Image, dict]:
    """
    Read DICOM series volumes with given paths.

    :param paths: Paths to DICOM volume files.
    :type paths: List[str]
    :param anonymize: Specify whether to hide PatientID and PatientName fields.
    :type anonymize: bool
    :return: Volume data in SimpleITK.Image format and dictionary with metadata.
    :rtype: Tuple[SimpleITK.Image, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        paths = ["/home/admin/work/volumes/vol_01.nrrd"]
        sitk_volume, meta = sly.volume.read_dicom_serie_volume(paths)
    """

    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    sitk_volume = reader.Execute()
    sitk_volume = _sitk_image_orient_ras(sitk_volume)
    dicom_tags = read_dicom_tags(paths[0], anonymize=anonymize)

    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(sitk_volume)
    meta = get_meta(
        sitk_volume.GetSize(),
        f_min_max.GetMinimum(),
        f_min_max.GetMaximum(),
        sitk_volume.GetSpacing(),
        sitk_volume.GetOrigin(),
        sitk_volume.GetDirection(),
        dicom_tags,
    )
    return sitk_volume, meta


def compose_ijk_2_world_mat(meta: dict) -> np.ndarray:
    """
    Transform 4x4 matrix from voxels to world coordinates.

    :param meta: Volume metadata.
    :type meta: dict
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        mat = sly.volume.volume.compose_ijk_2_world_mat(volume_meta)

        # Output:
        # [
        #     [   0.76171899    0.            0.         -194.23840308]
        #     [   0.            0.76171899    0.         -217.53840613]
        #     [   0.            0.            2.5        -347.75      ]
        #     [   0.            0.            0.            1.        ]
        # ]
    """

    try:
        spacing = meta["spacing"]
        origin = meta["origin"]
        directions = meta["directions"]
    except KeyError as e:
        raise IOError(
            f"Need the meta '{e}'' field to determine the mapping from voxels to world coordinates."
        )

    mat = np.eye(4)
    mat[:3, :3] = (np.array(directions).reshape(3, 3) * spacing).T
    mat[:3, 3] = origin
    return mat


def world_2_ijk_mat(ijk_2_world) -> np.ndarray:
    """
    Transform 4x4 matrix from world to voxels coordinates.

    :param ijk_2_world: 4x4 matrix.
    :type ijk_2_world: np.ndarray
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        mat = sly.volume.volume.world_2_ijk_mat(world_mat)

        # Output:
        # [
        #     [  1.3128201    0.           0.         255.00008013]
        #     [  0.           1.3128201    0.         285.58879251]
        #     [  0.           0.           0.4        139.1       ]
        #     [  0.           0.           0.           1.        ]
        # ]
    """

    return np.linalg.inv(ijk_2_world)


def get_meta(
    sitk_shape: tuple,
    min_intensity: float,
    max_intensity: float,
    spacing: tuple,
    origin: tuple,
    directions: tuple,
    dicom_tags: dict = {},
) -> dict:
    """
    Get normalized meta-data for a volume.

    :param sitk_shape: Tuple representing the shape of the volume in (x, y, z) dimensions.
    :type sitk_shape: tuple
    :param min_intensity: Minimum intensity value in the volume.
    :type min_intensity: float
    :param max_intensity: Maximum intensity value in the volume.
    :type max_intensity: float
    :param spacing: Tuple representing the spacing between voxels in (x, y, z) dimensions.
    :type spacing: tuple
    :param origin: Tuple representing the origin of the volume in (x, y, z) dimensions.
    :type origin: tuple
    :param directions: Tuple representing the direction matrix of the volume.
    :type directions: tuple
    :param dicom_tags: Dictionary containing additional DICOM tags for the volume meta-data.
    :type dicom_tags: dict, optional
    :return: Dictionary containing the normalized meta-data for the volume.
    :rtype: dict

    :Usage example:

     .. code-block:: python

        import SimpleITK as sitk
        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nrrd"

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(path)
        sitk_volume = reader.Execute()
        sitk_volume = _sitk_image_orient_ras(sitk_volume)
        dicom_tags = read_dicom_tags(paths[0], anonymize=anonymize)

        f_min_max = sitk.MinimumMaximumImageFilter()
        f_min_max.Execute(sitk_volume)
        meta = get_meta(
            sitk_volume.GetSize(),
            f_min_max.GetMinimum(),
            f_min_max.GetMaximum(),
            sitk_volume.GetSpacing(),
            sitk_volume.GetOrigin(),
            sitk_volume.GetDirection(),
            dicom_tags,
        )
    """

    # x = 1 - sagittal
    # y = 1 - coronal
    # z = 1 - axial
    volume_meta = normalize_volume_meta(
        {
            **dicom_tags,
            "channelsCount": 1,
            "rescaleSlope": 1,
            "rescaleIntercept": 0,
            "intensity": {
                "min": min_intensity,
                "max": max_intensity,
            },
            "dimensionsIJK": {
                "x": sitk_shape[0],
                "y": sitk_shape[1],
                "z": sitk_shape[2],
            },
            "ACS": "RAS",
            # instead of IJK2WorldMatrix field
            "spacing": spacing,
            "origin": origin,
            "directions": directions,
        }
    )
    return volume_meta


def inspect_nrrd_series(root_dir: str, logging: bool = True) -> List[str]:
    """
    Inspect a directory for NRRD series by recursively listing files with the ".nrrd" extension and returns a list of NRRD file paths found in the directory.

    :param root_dir: Directory to inspect for NRRD series.
    :type root_dir: str
    :param logging: Specify whether to print logging messages.
    :type logging: bool
    :return: List of NRRD file paths found in the given directory.
    :rtype: List[str]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/"
        nrrd_paths = sly.volume.inspect_nrrd_series(root_dir=path)
    """

    nrrd_paths = list_files_recursively(root_dir, [".nrrd"])
    if logging:
        logger.info(f"Total {len(nrrd_paths)} NRRD series in directory {root_dir}")
    return nrrd_paths


def read_nrrd_serie_volume(path: str) -> Tuple[sitk.Image, dict]:
    """
    Read NRRD volume with given path.

    :param path: Path to NRRD volume files.
    :type path: List[str]
    :return: Volume data in SimpleITK.Image format and dictionary with metadata.
    :rtype: Tuple[SimpleITK.Image, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nrrd"
        sitk_volume, meta = sly.volume.read_nrrd_serie_volume(path)
    """

    import SimpleITK as sitk

    # find custom NRRD loader in gitlab supervisely_py/-/blob/feature/import-volumes/plugins/import/volumes/src/loaders/nrrd.py
    reader = sitk.ImageFileReader()
    # reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(path)
    sitk_volume = reader.Execute()

    sitk_volume = _sitk_image_orient_ras(sitk_volume)
    f_min_max = sitk.MinimumMaximumImageFilter()
    f_min_max.Execute(sitk_volume)
    meta = get_meta(
        sitk_volume.GetSize(),
        f_min_max.GetMinimum(),
        f_min_max.GetMaximum(),
        sitk_volume.GetSpacing(),
        sitk_volume.GetOrigin(),
        sitk_volume.GetDirection(),
        {},
    )
    return sitk_volume, meta


def read_nrrd_serie_volume_np(paths: str) -> Tuple[np.ndarray, dict]:
    """
    Read NRRD volume with given path.

    :param paths: Path to NRRD volume file.
    :type paths: str
    :return: Volume data in NumPy array format and dictionary with metadata.
    :rtype: Tuple[np.ndarray, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nrrd"
        np_volume, meta = sly.volume.read_nrrd_serie_volume_np(path)
    """

    import SimpleITK as sitk

    sitk_volume, meta = read_nrrd_serie_volume(paths)
    volume_np = sitk.GetArrayFromImage(sitk_volume)
    volume_np = np.transpose(volume_np, (2, 1, 0))
    return volume_np, meta


def convert_nifti_to_nrrd(path: str) -> Tuple[np.ndarray, dict]:
    """Convert NIFTI volume to NRRD format.
    Volume automatically reordered to RAS orientation as closest to canonical.

    :param path: Path to NIFTI volume file.
    :type path: str
    :return: Volume data in NumPy array format and dictionary with metadata (NRRD header).
    :rtype: Tuple[np.ndarray, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nii"
        data, header = sly.volume.convert_nifti_to_nrrd(path)
    """

    import nibabel as nib  # pylint: disable=import-error

    nifti = nib.load(path)
    reordered_to_ras_nifti = nib.as_closest_canonical(nifti)
    data = reordered_to_ras_nifti.get_fdata()
    affine = reordered_to_ras_nifti.affine
    orientation = nib.aff2axcodes(affine)
    header = {
        "space": "".join(orientation),
        "space directions": affine.tolist(),
        "sizes": data.shape,
        "type": str(data.dtype),
        "dimension": len(data.shape),
    }
    return data, header


def convert_3d_nifti_to_nrrd(path: str) -> Tuple[np.ndarray, dict]:
    """Convert 3D NIFTI volume to NRRD format.
    Volume automatically reordered to RAS orientation as closest to canonical.

    :param path: Path to NIFTI volume file.
    :type path: str
    :return: Volume data in NumPy array format and dictionary with metadata (NRRD header).
    :rtype: Tuple[np.ndarray, dict]
    :Usage example:

     .. code-block:: python

        import supervisely as sly

        path = "/home/admin/work/volumes/vol_01.nii"
        data, header = sly.volume.convert_nifti_to_nrrd(path)
    """
    import SimpleITK as sitk

    nifti_image = sitk.ReadImage(path)
    nifti_image = _sitk_image_orient_ras(nifti_image)
    data = sitk.GetArrayFromImage(nifti_image)
    data = np.transpose(data, (2, 1, 0))

    direction = np.array(nifti_image.GetDirection()).reshape(3, 3)
    spacing = np.array(nifti_image.GetSpacing())
    origin = np.array(nifti_image.GetOrigin())

    space_directions = (direction.T * spacing[:, None]).tolist()

    header = {
        "dimension": 3,
        "space": "right-anterior-superior",
        "sizes": list(data.shape),
        "space directions": space_directions,
        "endian": "little",
        "encoding": "gzip",
        "space origin": origin,
    }
    return data, header


def is_nifti_file(path: str) -> bool:
    """Check if the file is a NIFTI file.

    :param filepath: Path to the file.
    :type filepath: str
    :return: True if the file is a NIFTI file, False otherwise.
    :rtype: bool
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "No module named nibabel. Please make sure that module is installed from pip and try again."
        )

    try:
        nib.load(path)
        return True
    except nib.filebasedimages.ImageFileError:
        return False


def convert_3d_geometry_to_mesh(
    geometry: Mask3D,
    spacing: tuple = (1.0, 1.0, 1.0),
    level: float = 0.5,
    apply_decimation: bool = False,
    decimation_fraction: float = 0.5,
    volume_meta: Optional[dict] = None,
) -> Trimesh:
    """
    Converts a 3D geometry (Mask3D) to a Trimesh mesh.

    :param geometry: The 3D geometry to convert.
    :type geometry: supervisely.geometry.mask_3d.Mask3D
    :param spacing: Voxel spacing in (x, y, z).
    :type spacing: tuple
    :param level: Isosurface value for marching cubes. Default is 0.5.
    :type level: float
    :param apply_decimation: Whether to simplify the mesh. Default is False.
    :type apply_decimation: bool
    :param decimation_fraction: Fraction of faces to keep if decimation is applied. Default is 0.5.
    :type decimation_fraction: float
    :param volume_meta: Metadata of the volume. Used for mesh alignment if geometry lacks specific fields. Default is None.
    :type volume_meta: dict, optional
    :return: The resulting Trimesh mesh.
    :rtype: trimesh.Trimesh

    :Usage example:

        .. code-block:: python

            volume_header = nrrd.read_header("path/to/volume.nrrd")
            mask3d = Mask3D.create_from_file("path/to/mask3d")
            mesh = convert_3d_geometry_to_mesh(mask3d, spacing=(1.0, 1.0, 1.0), level=0.7, apply_decimation=True, volume_meta=volume_header)
    """
    from skimage import measure

    if volume_meta is None:
        volume_meta = {}

    space_directions = geometry.space_directions or volume_meta.get("space directions")
    space_origin = geometry.space_origin or volume_meta.get("space origin")

    verts, faces, normals, _ = measure.marching_cubes(geometry.data, level=level, spacing=spacing)
    mesh = Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)

    if apply_decimation and 0 < decimation_fraction < 1:
        mesh = mesh.simplify_quadric_decimation(int(len(mesh.faces) * decimation_fraction))

    if space_directions is not None and space_origin is not None:
        header = {
            "space directions": space_directions,
            "space origin": space_origin,
        }
        align_mesh_to_volume(mesh, header)

    # flip x and y axes to match initial mask orientation
    mesh.apply_transform(np.diag([-1, -1, 1, 1]))

    mesh.fix_normals()

    return mesh


def export_3d_as_mesh(geometry: Mask3D, output_path: str, **kwargs):
    """
    Exports the 3D mesh representation of the object to a file in either STL or OBJ format.

    :param geometry: The 3D geometry to be exported.
    :type geometry: supervisely.geometry.mask_3d.Mask3D
    :param output_path: The path to the output file. Must have a ".stl" or ".obj" extension.
    :type output_path: str
    :param kwargs: Additional keyword arguments for mesh generation. Supported keys:
        - spacing (tuple): Voxel spacing in (x, y, z). By default the value will be taken from geometry meta.
        - level (float): Isosurface value for marching cubes. Default is 0.5.
        - apply_decimation (bool): Whether to simplify the mesh. Default is False.
        - decimation_fraction (float): Fraction of faces to keep if decimation is applied. Default is 0.5.
        - volume_meta (dict): Metadata of the volume. Used for mesh alignment if geometry lacks specific fields. Default is None.
    :return: None

    :Usage example:

        .. code-block:: python

        mask3d_path = "path/to/mask3d"
        mask3d = Mask3D.create_from_file(mask3d_path)

        mask3d.export_3d_as_mesh(mask3d, "output.stl", spacing=(1.0, 1.0, 1.0), level=0.7, apply_decimation=True)
    """

    if get_file_ext(output_path).lower() not in [".stl", ".obj"]:
        raise ValueError('File extension must be either ".stl" or ".obj"')

    mesh = convert_3d_geometry_to_mesh(geometry, **kwargs)
    mesh.export(output_path)


def align_mesh_to_volume(mesh: Trimesh, volume_header: dict) -> None:
    """
    Transforms the given mesh in-place using spatial information from an NRRD header.
    The mesh will be tranformed to match the coordinate system defined in the header.

    :param mesh: The mesh object to be transformed. The transformation is applied in-place.
    :type mesh: Trimesh
    :param volume_header: The NRRD header containing spatial metadata, including "space directions",
        "space origin", and "space". Field "space" should be in the format of
        "right-anterior-superior", "left-anterior-superior", etc.
    :type volume_header: dict
    :returns: None
    :rtype: None
    """
    from supervisely.geometry.constants import SPACE_ORIGIN
    from supervisely.geometry.mask_3d import PointVolume

    if isinstance(volume_header["space origin"], PointVolume):
        volume_header["space origin"] = volume_header["space origin"].to_json()[SPACE_ORIGIN]
    transform_mat = matrix_from_nrrd_header(volume_header)
    mesh.apply_transform(transform_mat)
