"""Volume import through ImportManager: DICOM series, NRRD files and both in one input.

Runs offline: the input is a local folder of synthetic CT slices and NRRD volumes, and the upload
goes to a fake ``api.volume`` that records what would be sent.
"""

import os
from types import SimpleNamespace

import nrrd
import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid  # noqa: E402

import supervisely.convert.converter as converter_module  # noqa: E402
from supervisely.convert.converter import ImportManager  # noqa: E402
from supervisely.convert.volume.dicom import dicom_converter as dicom_module  # noqa: E402
from supervisely.convert.volume.dicom.dicom_converter import DICOMConverter  # noqa: E402
from supervisely.convert.volume.volume_converter import VolumeConverter  # noqa: E402
from supervisely.project.project_type import ProjectType  # noqa: E402


def _write_series(out_dir, n, rows=16, cols=16, with_ext=True, prefix="IMG"):
    os.makedirs(out_dir, exist_ok=True)
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    for i in range(n):
        path = os.path.join(out_dir, f"{prefix}-{i + 1:05d}" + (".dcm" if with_ext else ""))
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientName = "Unit^Test"
        ds.PatientID = "UNIT-001"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SeriesNumber = 1
        ds.InstanceNumber = i + 1
        ds.Modality = "CT"
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * 2.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.PixelData = np.full((rows, cols), 100 * i, dtype=np.int16).tobytes()
        ds.save_as(path, enforce_file_format=True)
    return series_uid


def _write_nrrd(path, shape=(10, 8, 6)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.zeros(shape, dtype=np.int16)
    data[2:5, 2:5, 2:4] = 300
    nrrd.write(
        path,
        data,
        {
            "space": "left-posterior-superior",
            "space directions": np.diag([1.0, 1.0, 2.0]),
            "space origin": [0.0, 0.0, 0.0],
        },
    )


class _FakeVolumeApi:
    def __init__(self, existing_names=()):
        self.calls = []
        self._existing = list(existing_names)

    def get_list(self, dataset_id):
        return [SimpleNamespace(name=n) for n in self._existing]

    def upload_dicom_serie_paths(self, dataset_id, name, paths, log_progress=True, anonymize=True):
        self.calls.append(("dicom", name, sorted(os.path.basename(p) for p in paths), anonymize))

    def upload_nrrd_series_paths(self, dataset_id, names, paths, progress_cb=None, log_progress=True):
        for name, path in zip(names, paths):
            self.calls.append(("nrrd", name, os.path.basename(path), None))
        return [SimpleNamespace(id=i) for i in range(len(names))]


class _FakeApi:
    def __init__(self, existing_names=()):
        self.volume = _FakeVolumeApi(existing_names)


@pytest.fixture
def volume_env(tmp_path, monkeypatch):
    """ImportManager for volumes on a local input dir, with no Supervisely instance behind it."""
    monkeypatch.setenv("SLY_APP_DATA_DIR", str(tmp_path / "app_data"))
    monkeypatch.setenv("TEAM_ID", "1")
    monkeypatch.setattr(converter_module.Api, "from_env", classmethod(lambda cls: object()))
    warnings = []
    original_warning = dicom_module.logger.warning

    def capture_warning(msg, *args, **kwargs):
        warnings.append(str(msg))
        return original_warning(msg, *args, **kwargs)

    monkeypatch.setattr(dicom_module.logger, "warning", capture_warning)
    src = tmp_path / "input"
    src.mkdir()
    return src, warnings


def _upload(converter, existing_names=()):
    api = _FakeApi(existing_names)
    converter.upload_dataset(api, dataset_id=1)
    return api.volume.calls


def test_dicom_and_nrrd_in_one_folder_import_both(volume_env):
    """The ct_parity layout: series in subfolders, one NRRD at the top; nothing is dropped."""
    src, warnings = volume_env
    uid1 = _write_series(str(src / "studyA" / "series1"), 4)
    uid2 = _write_series(str(src / "studyA" / "series2"), 3, with_ext=False)
    _write_nrrd(str(src / "extra_volume.nrrd"))

    importer = ImportManager(str(src), ProjectType.VOLUMES)

    assert isinstance(importer.converter, DICOMConverter)
    assert importer.converter.items_count == 3
    calls = _upload(importer.converter)
    # series first (in os.walk order), then the NRRD files, like the "Import DICOM Volumes" app
    assert sorted(c[:2] for c in calls[:2]) == sorted(
        [("dicom", f"{uid1}.nrrd"), ("dicom", f"{uid2}.nrrd")]
    )
    assert calls[2][:3] == ("nrrd", "extra_volume.nrrd", "extra_volume.nrrd")
    assert all(c[3] is True for c in calls if c[0] == "dicom")
    assert warnings == []


def test_dicom_only_import_is_unchanged(volume_env):
    src, warnings = volume_env
    uid = _write_series(str(src / "series"), 5)

    importer = ImportManager(str(src), ProjectType.VOLUMES)

    assert isinstance(importer.converter, DICOMConverter)
    items = importer.converter.get_items()
    assert len(items) == 1 and isinstance(items[0], DICOMConverter.Item)
    assert items[0].volume_meta["dimensionsIJK"] == {"x": 16, "y": 16, "z": 5}
    assert _upload(importer.converter) == [
        ("dicom", f"{uid}.nrrd", [f"IMG-{i:05d}.dcm" for i in range(1, 6)], True)
    ]
    assert warnings == []


def test_nrrd_only_import_still_goes_through_the_generic_volume_path(volume_env):
    src, _ = volume_env
    _write_nrrd(str(src / "a.nrrd"))
    _write_nrrd(str(src / "sub" / "b.nrrd"))

    importer = ImportManager(str(src), ProjectType.VOLUMES)

    assert type(importer.converter) is VolumeConverter
    assert sorted(item.name for item in importer.converter.get_items()) == ["a.nrrd", "b.nrrd"]


def test_anonymization_is_on_by_default_and_can_be_switched_off(volume_env):
    src, _ = volume_env
    _write_series(str(src / "series"), 3)
    assert dicom_module.ANONYMIZE_DICOM_BY_DEFAULT is True

    default = ImportManager(str(src), ProjectType.VOLUMES)
    assert default.converter.anonymize is True
    assert [c[3] for c in _upload(default.converter)] == [True]

    kept = ImportManager(str(src), ProjectType.VOLUMES, anonymize_dicom=False)
    assert kept.converter.anonymize is False
    assert [c[3] for c in _upload(kept.converter)] == [False]

    explicit = ImportManager(str(src), ProjectType.VOLUMES, anonymize_dicom=True)
    assert [c[3] for c in _upload(explicit.converter)] == [True]


def test_anonymize_dicom_is_ignored_for_non_dicom_input(volume_env):
    src, _ = volume_env
    _write_nrrd(str(src / "a.nrrd"))

    importer = ImportManager(str(src), ProjectType.VOLUMES, anonymize_dicom=False)

    assert type(importer.converter) is VolumeConverter


def test_nrrd_named_like_a_series_volume_gets_a_free_name(volume_env):
    src, _ = volume_env
    uid = _write_series(str(src / "series"), 3)
    _write_nrrd(str(src / f"{uid}.nrrd"))
    _write_nrrd(str(src / "taken.nrrd"))

    importer = ImportManager(str(src), ProjectType.VOLUMES)
    names = [c[1] for c in _upload(importer.converter, existing_names=["taken.nrrd"])]

    assert names == [f"{uid}.nrrd", f"{uid}_01.nrrd", "taken_01.nrrd"]


def test_uppercase_nrrd_extension_is_imported_with_lowercase_name(volume_env):
    src, _ = volume_env
    _write_series(str(src / "series"), 3)
    _write_nrrd(str(src / "UPPER.NRRD"))

    importer = ImportManager(str(src), ProjectType.VOLUMES)
    calls = _upload(importer.converter)

    assert calls[-1][:3] == ("nrrd", "UPPER.nrrd", "UPPER.nrrd")


def test_files_that_are_not_series_or_nrrd_are_reported(volume_env):
    src, warnings = volume_env
    _write_series(str(src / "series"), 3)
    _write_nrrd(str(src / "vol.nrrd"))
    (src / "notes.txt").write_text("not a volume")
    (src / "series" / "readme.md").write_text("not a slice")

    importer = ImportManager(str(src), ProjectType.VOLUMES)

    assert importer.converter.items_count == 2
    skipped = [w for w in warnings if "will not be imported" in w]
    assert len(skipped) == 1
    assert "2 file(s)" in skipped[0]
    assert "notes.txt" in skipped[0] and os.path.join("series", "readme.md") in skipped[0]


def test_unreadable_nrrd_next_to_dicom_is_skipped_not_fatal(volume_env):
    src, warnings = volume_env
    uid = _write_series(str(src / "series"), 3)
    (src / "broken.nrrd").write_bytes(b"NRRD0004\nthis is not a valid header\n")

    importer = ImportManager(str(src), ProjectType.VOLUMES)

    assert [c[1] for c in _upload(importer.converter)] == [f"{uid}.nrrd"]
    assert any("broken.nrrd" in w and "skipped" in w for w in warnings)
