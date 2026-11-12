"""DICOM import on pydicom 2 (Python < 3.10) and pydicom 3 (Python >= 3.10).

pydicom 3 removed ``read_file`` (every DICOM was "not a DICOM file" for the medical 2D converter) and
changed ``Dataset.decompress()``: YBR colour data is converted to RGB and the SOP Instance UID is
replaced. These tests build small DICOM files with whichever pydicom is installed and check the
results pydicom 2 has always produced, so the same assertions hold on both versions.
"""

import io
import os
import re
from pathlib import Path

import nrrd
import numpy as np
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    RLELossless,
    SecondaryCaptureImageStorage,
)

import supervisely.convert.converter as converter_module
from supervisely.convert.converter import ImportManager
from supervisely.convert.image.medical2d import medical2d_helper
from supervisely.convert.volume.dicom import dicom_helper
from supervisely.project.project_type import ProjectType

PYDICOM_3 = int(pydicom.__version__.split(".", 1)[0]) >= 3
STUDY_UID = "1.2.826.0.1.3680043.8.498.11"
SERIES_UID = "1.2.826.0.1.3680043.8.498.12"
ROOT = Path(__file__).resolve().parents[2]


def _dataset(path, sop_class, transfer_syntax, sop_uid):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = transfer_syntax
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.StudyID = "S1"
    ds.Modality = "CT"
    ds.Manufacturer = "Synth"
    ds.PatientName = "Doe^John"
    ds.PatientID = "P1"
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.7]
    ds.SliceThickness = 1
    return ds


def _save(ds, path):
    if PYDICOM_3:
        pydicom.dcmwrite(str(path), ds, enforce_file_format=True)
    else:
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(str(path), write_like_original=False)


def _mono_dataset(path, arr, sop_uid, frames=None):
    ds = _dataset(path, CTImageStorage, ExplicitVRLittleEndian, sop_uid)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = arr.shape[-2:]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if arr.dtype == np.int16 else 0
    if frames:
        ds.NumberOfFrames = frames
    ds.PixelData = arr.tobytes()
    return ds


def write_ct(path, arr, frames=None, sop_uid="1.2.826.0.1.3680043.8.498.21"):
    _save(_mono_dataset(path, arr, sop_uid, frames), path)


def write_ct_rle(path, arr, sop_uid="1.2.826.0.1.3680043.8.498.22"):
    ds = _mono_dataset(path, arr, sop_uid)
    ds.compress(RLELossless, arr)
    # compress() gives the dataset a new SOP Instance UID on pydicom 3; keep ours
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = sop_uid
    _save(ds, path)


def jpeg_ybr422(rgb):
    """Baseline JPEG with 4:2:2 chroma subsampling, as ultrasound devices write it."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95, subsampling=1)
    return buf.getvalue()


def luma(jpeg_bytes):
    """Y channel of a JPEG decoded without colour conversion (what pydicom 2 handed over)."""
    img = Image.open(io.BytesIO(jpeg_bytes))
    img.draft("YCbCr", img.size)
    return np.asarray(img)[..., 0]


def write_us_ybr422(path, rgb_frames, sop_uid="1.2.826.0.1.3680043.8.498.23"):
    ds = _dataset(path, SecondaryCaptureImageStorage, JPEGBaseline8Bit, sop_uid)
    ds.Modality = "US"
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "YBR_FULL_422"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = rgb_frames[0].shape[:2]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    if len(rgb_frames) > 1:
        ds.NumberOfFrames = len(rgb_frames)
    blobs = [jpeg_ybr422(f) for f in rgb_frames]
    ds.PixelData = encapsulate(blobs)
    ds["PixelData"].is_undefined_length = True
    ds["PixelData"].VR = "OB"
    _save(ds, path)
    return [luma(b) for b in blobs]


def write_native_ybr422(path, y, cb, cr, sop_uid="1.2.826.0.1.3680043.8.498.24"):
    """Uncompressed YBR_FULL_422: two luma samples share one Cb/Cr pair (Y1 Y2 Cb Cr per pixel pair)."""
    ds = _dataset(path, SecondaryCaptureImageStorage, ExplicitVRLittleEndian, sop_uid)
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "YBR_FULL_422"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = y.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    packed = np.empty((y.shape[0], y.shape[1] // 2, 4), dtype=np.uint8)
    packed[..., 0] = y[:, 0::2]
    packed[..., 1] = y[:, 1::2]
    packed[..., 2] = cb
    packed[..., 3] = cr
    ds.PixelData = packed.tobytes()
    _save(ds, path)


def _gradient(h=24, w=32):
    yy, xx = np.mgrid[0:h, 0:w]
    return yy, xx


def _rgb(h=24, w=32):
    yy, xx = _gradient(h, w)
    return np.stack([xx * 8, yy * 10, (xx + yy) * 4], axis=-1).astype(np.uint8)


def _single_frame_view(arr):
    """medical2d stores a single-frame slice rotated by 270 degrees and flipped left-right.

    The rotation goes through ImageRotator, which pads the slice by one pixel (a 64x64 slice comes
    out 65x65 in both main-import and the old import-dicom-studies app), so it is reproduced with
    the same SDK functions instead of np.rot90.
    """
    from supervisely.imaging import image as sly_image

    return sly_image.fliplr(sly_image.rotate(img=arr, degrees_angle=270))


def test_no_pydicom_api_removed_in_v3_is_used():
    removed = re.compile(
        r"\bread_file\s*\(|\bwrite_file\s*\(|write_like_original|pydicom\.filereader\.read_file"
        r"|pydicom\.filewriter\.write_file|\.fix_meta_info\(|overlay_data_handlers|PersonName3"
    )
    offenders = []
    for path in (ROOT / "supervisely").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "pydicom" not in text and "dicom" not in path.name:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if removed.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert offenders == []


def test_is_dicom_file(tmp_path):
    dcm = tmp_path / "ct.dcm"
    yy, xx = _gradient()
    write_ct(dcm, (yy * 40 - xx * 7).astype(np.int16))
    not_dcm = tmp_path / "notes.dcm"
    not_dcm.write_bytes(b"\0" * 128 + b"not a dicom file at all")

    assert medical2d_helper.is_dicom_file(str(dcm)) is True
    assert medical2d_helper.is_dicom_file(str(not_dcm)) is False


def test_single_frame_ct_to_nrrd(tmp_path):
    pytest.importorskip("skimage")  # supervisely.imaging.image.rotate imports it
    yy, xx = _gradient()
    arr = (yy * 40 - xx * 7 - 300).astype(np.int16)
    src = tmp_path / "ct.dcm"
    write_ct(src, arr)

    paths, names, group_tags, meta = medical2d_helper.convert_dcm_to_nrrd(
        str(src), str(tmp_path / "converted")
    )

    assert names == ["ct.dcm.nrrd"]
    data, header = nrrd.read(paths[0])
    assert data.dtype == np.int16
    assert np.array_equal(data, _single_frame_view(arr))
    assert header["space"] == "right-anterior-superior"
    assert {t["name"]: t["value"] for t in group_tags} == {
        "StudyInstanceUID": STUDY_UID,
        "StudyID": "S1",
        "SeriesInstanceUID": SERIES_UID,
        "Manufacturer": "Synth",
        "Modality": "CT",
    }
    assert meta["Patient's Name"] == "anonymized"
    assert meta["Patient ID"] == "anonymized"
    assert meta["SOP Instance UID"] == "1.2.826.0.1.3680043.8.498.21"
    assert meta["Photometric Interpretation"] == "MONOCHROME2"


def test_multi_frame_ct_to_nrrd(tmp_path):
    frames = np.random.RandomState(0).randint(0, 4000, (3, 20, 16)).astype(np.uint16)
    src = tmp_path / "multi.dcm"
    write_ct(src, frames, frames=3)

    paths, names, _, meta = medical2d_helper.convert_dcm_to_nrrd(
        str(src), str(tmp_path / "converted")
    )

    assert names == ["1_multi.dcm.nrrd", "2_multi.dcm.nrrd", "3_multi.dcm.nrrd"]
    for path, frame in zip(paths, frames):
        data, _ = nrrd.read(path)
        assert np.array_equal(data, frame.T)
    assert meta["Number of Frames"] == "3"


def test_compressed_ct_keeps_sop_instance_uid(tmp_path):
    """pydicom 3 decompress() makes a new SOP Instance UID by default; the image meta must keep the file's."""
    pytest.importorskip("skimage")
    yy, xx = _gradient()
    arr = (yy * 50 + xx).astype(np.uint16)
    src = tmp_path / "rle.dcm"
    write_ct_rle(src, arr, sop_uid="1.2.826.0.1.3680043.8.498.99")

    paths, _, _, meta = medical2d_helper.convert_dcm_to_nrrd(str(src), str(tmp_path / "converted"))

    data, _ = nrrd.read(paths[0])
    assert np.array_equal(data, _single_frame_view(arr))
    assert meta["SOP Instance UID"] == "1.2.826.0.1.3680043.8.498.99"


def test_ybr_full_422_jpeg_becomes_monochrome(tmp_path):
    """Ultrasound-style JPEG (YBR_FULL_422) is imported as its luma channel, as on pydicom 2.

    On pydicom 3 decompress() turned it into RGB, so it was not converted and the import failed.
    """
    pytest.importorskip("skimage")
    src = tmp_path / "us.dcm"
    (expected,) = write_us_ybr422(src, [_rgb()])

    paths, names, _, meta = medical2d_helper.convert_dcm_to_nrrd(str(src), str(tmp_path / "converted"))

    assert names == ["us.dcm.nrrd"]
    data, _ = nrrd.read(paths[0])
    assert data.dtype == np.uint8
    assert np.array_equal(data, _single_frame_view(expected))
    assert meta["Photometric Interpretation"] == "MONOCHROME2"
    assert meta["Samples per Pixel"] == "1"
    assert meta["SOP Instance UID"] == "1.2.826.0.1.3680043.8.498.23"


def test_multi_frame_ybr_full_422_jpeg_becomes_monochrome(tmp_path):
    src = tmp_path / "us_multi.dcm"
    rgb = _rgb()
    expected = write_us_ybr422(src, [rgb, 255 - rgb])

    paths, names, _, _ = medical2d_helper.convert_dcm_to_nrrd(str(src), str(tmp_path / "converted"))

    assert names == ["1_us_multi.dcm.nrrd", "2_us_multi.dcm.nrrd"]
    for path, frame in zip(paths, expected):
        data, _ = nrrd.read(path)
        assert np.array_equal(data, frame.T)


def test_volume_helper_rewrites_ybr_full_422_as_monochrome(tmp_path):
    """The DICOM volume converter rewrites YBR_FULL_422 files as MONOCHROME2 before SimpleITK reads them."""
    src = tmp_path / "us.dcm"
    rgb = _rgb()
    expected = write_us_ybr422(src, [rgb, 255 - rgb])

    dicom_helper.read_and_convert_to_monochrome2(str(src))

    ds = pydicom.dcmread(str(src))
    assert ds.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert ds.PhotometricInterpretation == "MONOCHROME2"
    assert ds.SamplesPerPixel == 1
    assert ds.SOPInstanceUID == "1.2.826.0.1.3680043.8.498.23"
    assert np.array_equal(ds.pixel_array, np.stack(expected))


def test_volume_helper_rewrites_uncompressed_ybr_full_422_as_monochrome(tmp_path):
    src = tmp_path / "native.dcm"
    yy, xx = _gradient()
    y = (yy * 9 + xx * 2).astype(np.uint8)
    cb = np.full((24, 16), 90, dtype=np.uint8)
    cr = np.full((24, 16), 170, dtype=np.uint8)
    write_native_ybr422(src, y, cb, cr)

    dicom_helper.read_and_convert_to_monochrome2(str(src))

    ds = pydicom.dcmread(str(src))
    assert ds.PhotometricInterpretation == "MONOCHROME2"
    assert ds.SOPInstanceUID == "1.2.826.0.1.3680043.8.498.24"
    assert np.array_equal(ds.pixel_array, y)


def test_volume_helper_leaves_monochrome_files_alone(tmp_path):
    src = tmp_path / "rle.dcm"
    yy, xx = _gradient()
    write_ct_rle(src, (yy * 50 + xx).astype(np.uint16))
    before = src.read_bytes()

    dicom_helper.read_and_convert_to_monochrome2(str(src))

    assert src.read_bytes() == before


@pytest.fixture
def import_env(tmp_path, monkeypatch):
    """ImportManager on a local input dir, with no Supervisely instance behind it."""
    monkeypatch.setenv("SLY_APP_DATA_DIR", str(tmp_path / "app_data"))
    monkeypatch.setenv("TEAM_ID", "1")
    monkeypatch.setattr(converter_module.Api, "from_env", classmethod(lambda cls: object()))
    src = tmp_path / "input"
    src.mkdir()
    return src


def test_import_manager_detects_dicom_studies_in_images_project(import_env):
    """The main-import path: DICOM files in an images project are found by the medical 2D converter."""
    pytest.importorskip("skimage")
    src = import_env
    yy, xx = _gradient()
    for folder, name, uid in [
        ("research_1", "a.dcm", "1.2.826.0.1.3680043.8.498.31"),
        ("research_1", "b.dcm", "1.2.826.0.1.3680043.8.498.32"),
        ("research_2", "c.dcm", "1.2.826.0.1.3680043.8.498.33"),
    ]:
        os.makedirs(src / folder, exist_ok=True)
        write_ct(src / folder / name, (yy * 40 - xx * 7).astype(np.int16), sop_uid=uid)

    importer = ImportManager(str(src), ProjectType.IMAGES)

    converter = importer.converter
    assert str(converter) == "medical_2d"
    assert converter.items_count == 3
    assert sorted(os.path.basename(item.path) for item in converter.get_items()) == [
        "a.dcm.nrrd",
        "b.dcm.nrrd",
        "c.dcm.nrrd",
    ]
    assert converter._group_tag_names["StudyInstanceUID"] == 3
    assert [t.name for t in converter.get_meta().tag_metas] == [
        "StudyInstanceUID",
        "StudyID",
        "SeriesInstanceUID",
        "Manufacturer",
        "Modality",
    ]
