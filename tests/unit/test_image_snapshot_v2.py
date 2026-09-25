# coding: utf-8
"""Round-trip tests for the Parquet image snapshot schemas v2.0.0 and v2.1.0.

The point of the schema is that a restore cannot tell the two formats apart: both
``Project.build_snapshot()`` and the pickle backup are read by
``Project._read_snapshot()`` into the same six-part payload, and everything after that
is one code path. So these tests build a snapshot from a stubbed API, read it back, and
assert the payload matches what the pickle format would have carried - field by field,
including the empty forms (``{}`` for meta, ``[]`` for tags) that the upload path
iterates without checking.
"""

import io
import os
import pickle
import tempfile
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow", reason="image snapshot v2.0.0 needs the 'versioning' extra")

from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.annotation.tag_meta import TagMeta, TagValueType
from supervisely.annotation.tag_meta_collection import TagMetaCollection
from supervisely.api.dataset_api import DatasetInfo
from supervisely.api.entity_annotation.figure_api import FigureInfo
from supervisely.api.image_api import ImageInfo
from supervisely.api.project_api import ProjectInfo
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.rectangle import Rectangle
from supervisely.project.project import Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.versioning.common import (
    IMAGE_SCHEMA_VERSION_V2,
    IMAGE_SCHEMA_VERSION_V2_1,
)

PROJECT_ID = 777
ALPHA_FIGURE_ID = 9003


def _make(cls, **values):
    """Build a NamedTuple with only the fields the test cares about set."""
    return cls(**{field: values.get(field) for field in cls._fields})


def _meta() -> ProjectMeta:
    # Colors are explicit: ObjClass/TagMeta pick a random one when none is given, and
    # two calls to this helper have to produce an identical meta for the comparisons below.
    return ProjectMeta(
        obj_classes=ObjClassCollection(
            [
                ObjClass("car", Rectangle, color=[255, 0, 0], sly_id=11),
                ObjClass("road", Polygon, color=[0, 255, 0], sly_id=12),
                ObjClass("blob", AlphaMask, color=[0, 0, 255], sly_id=13),
            ]
        ),
        tag_metas=TagMetaCollection(
            [TagMeta("reviewed", TagValueType.NONE, color=[128, 128, 0], sly_id=21)]
        ),
    )


def _project_info() -> ProjectInfo:
    return _make(
        ProjectInfo,
        id=PROJECT_ID,
        name="cityscape",
        description="a project",
        readme="# readme",
        type="images",
        custom_data={"origin": "test"},
        settings={"allowDuplicateTags": False},
        version={"id": 5, "version": 4},
    )


def _datasets():
    root = _make(
        DatasetInfo, id=1, name="train", description="root ds", parent_id=None, custom_data={}
    )
    nested = _make(
        DatasetInfo,
        id=2,
        name="night",
        description=None,
        parent_id=1,
        custom_data={"weather": "rain"},
    )
    return root, nested


def _images():
    return {
        1: [
            _make(
                ImageInfo,
                id=101,
                dataset_id=1,
                name="a.jpg",
                hash="hash-a",
                link=None,
                mime="image/jpeg",
                ext="jpeg",
                # The API returns this as a JSON string for some images; the schema has
                # to swallow that rather than fail the whole snapshot.
                size="1024",
                width=800,
                height=600,
                created_at="2026-01-01T00:00:00.000Z",
                updated_at="2026-01-02T00:00:00.000Z",
                created_by=42,
                meta={"custom-sort": "3"},
                tags=[{"tagId": 21, "value": None}],
            ),
            _make(
                ImageInfo,
                id=102,
                dataset_id=1,
                name="b.jpg",
                hash=None,
                link="https://example.com/b.jpg",
                meta={},
                tags=[],
            ),
        ],
        2: [
            _make(
                ImageInfo,
                id=201,
                dataset_id=2,
                name="c.jpg",
                hash="hash-c",
                meta={},
                tags=[],
                parent_id=101,
            ),
        ],
    }


def _figures():
    return {
        101: [
            _make(
                FigureInfo,
                id=9001,
                entity_id=101,
                dataset_id=1,
                class_id=11,
                geometry_type=Rectangle.geometry_name(),
                geometry={"points": {"exterior": [[0, 0], [10, 10]], "interior": []}},
                geometry_meta={},
                meta={"note": "keep"},
                tags=[{"tagId": 21, "value": None}],
                priority=1,
                custom_data={"source": "model"},
                created_at="2026-03-01T00:00:00.000Z",
                updated_at="2026-03-02T00:00:00.000Z",
            ),
            _make(
                FigureInfo,
                id=ALPHA_FIGURE_ID,
                entity_id=101,
                dataset_id=1,
                class_id=13,
                geometry_type=AlphaMask.geometry_name(),
                geometry=None,
                geometry_meta={},
                meta={},
                tags=[],
            ),
        ],
        201: [
            _make(
                FigureInfo,
                id=9002,
                entity_id=201,
                dataset_id=2,
                class_id=12,
                geometry_type=Polygon.geometry_name(),
                geometry={"points": {"exterior": [[1, 1], [5, 1], [5, 5]], "interior": []}},
                geometry_meta={},
                meta={},
                tags=[],
            )
        ],
    }


ALPHA_GEOMETRY = {"bitmap": {"origin": [0, 0], "data": "base64-payload"}}


class _FakeFigureApi:
    def __init__(self, figures):
        self._figures = figures

    def download(self, dataset_id, image_ids, integer_coords=False):
        return {
            image_id: self._figures[image_id]
            for image_id in image_ids
            if image_id in self._figures
        }

    def download_geometries_batch(self, figure_ids):
        return [ALPHA_GEOMETRY for _ in figure_ids]


class _FakeAnnotationApi:
    """annotations.bulk.info: where the snapshot reads figure tags from."""

    def __init__(self, figures):
        self._figures = figures

    def _figure_tags_batch(self, dataset_id, image_ids):
        return {
            figure.id: figure.tags
            for image_id in image_ids
            for figure in self._figures.get(image_id, [])
            if figure.tags
        }


class _FakeImageApi:
    def __init__(self, images, figures):
        self._images = images
        self.figure = _FakeFigureApi(figures)

    def get_list(self, dataset_id):
        return self._images.get(dataset_id, [])

    def get_list_generator(self, dataset_id, batch_size=None, force_metadata_for_links=False):
        # Recorded, because the generator defaults this off where `get_list` defaults it on,
        # and a link-backed image listed without it has no width, height, mime or size.
        self.forced_metadata_for_links = force_metadata_for_links
        images = self._images.get(dataset_id, [])
        step = batch_size or len(images) or 1
        for start in range(0, len(images), step):
            yield images[start : start + step]


class _FakeDatasetApi:
    def __init__(self, datasets):
        self._datasets = datasets

    def get_list(self, project_id, recursive=False, include_custom_data=False, filters=None):
        return list(self._datasets)

    def tree(self, project_id):
        root, nested = self._datasets
        yield [], root
        yield [root.name], nested


class _FakeProjectApi:
    def __init__(self, project_info, meta):
        self._project_info = project_info
        self._meta = meta

    def get_info_by_id(self, project_id):
        return self._project_info

    def get_meta(self, project_id, with_settings=False):
        return self._meta.to_json()


class _FakeApi:
    def __init__(self):
        self.project = _FakeProjectApi(_project_info(), _meta())
        self.dataset = _FakeDatasetApi(_datasets())
        self.image = _FakeImageApi(_images(), _figures())
        self.annotation = _FakeAnnotationApi(_figures())


@pytest.fixture(scope="module")
def payload():
    snapshot = Project.build_snapshot(
        _FakeApi(),
        project_id=PROJECT_ID,
        batch_size=10,
        log_progress=False,
        schema_version=IMAGE_SCHEMA_VERSION_V2,
    )
    return Project._read_snapshot(snapshot)


def test_project_info_and_meta_survive(payload):
    project_info, meta = payload[0], payload[1]
    assert project_info.name == "cityscape"
    assert project_info.readme == "# readme"
    assert project_info.custom_data == {"origin": "test"}
    assert project_info.settings == {"allowDuplicateTags": False}
    assert project_info.version == {"id": 5, "version": 4}
    assert [c.name for c in meta.obj_classes] == ["car", "road", "blob"]
    assert [c.sly_id for c in meta.obj_classes] == [11, 12, 13]
    assert [t.name for t in meta.tag_metas] == ["reviewed"]


def test_dataset_tree_survives(payload):
    dataset_infos = payload[2]
    by_id = {ds.id: ds for ds in dataset_infos}
    assert set(by_id) == {1, 2}
    assert by_id[1].name == "train"
    assert by_id[1].description == "root ds"
    assert by_id[1].parent_id is None
    assert by_id[2].parent_id == 1
    assert by_id[2].custom_data == {"weather": "rain"}


def test_image_fields_survive(payload):
    image_infos = payload[3]
    by_id = {img.id: img for img in image_infos}
    assert set(by_id) == {101, 102, 201}

    first = by_id[101]
    assert first.name == "a.jpg"
    assert first.dataset_id == 1
    assert first.hash == "hash-a"
    assert first.link is None
    assert first.mime == "image/jpeg"
    assert first.ext == "jpeg"
    assert first.width == 800
    assert first.height == 600
    assert first.created_at == "2026-01-01T00:00:00.000Z"
    assert first.created_by == 42
    assert first.meta == {"custom-sort": "3"}
    assert first.tags == [{"tagId": 21, "value": None}]
    # A numeric field that arrived as a string is still stored, as a number.
    assert first.size == 1024

    linked = by_id[102]
    assert linked.link == "https://example.com/b.jpg"
    assert linked.hash is None
    # The upload path passes meta straight to the API and iterates tags, so the empty
    # forms must come back as {} and [], never as None.
    assert linked.meta == {}
    assert linked.tags == []

    assert by_id[201].parent_id == 101


def test_figures_are_grouped_by_image_and_keep_their_fields(payload):
    figures = payload[4]
    assert set(figures) == {101, 201}
    assert len(figures[101]) == 2

    rectangle = next(f for f in figures[101] if f.id == 9001)
    assert rectangle.entity_id == 101
    assert rectangle.dataset_id == 1
    assert rectangle.class_id == 11
    assert rectangle.geometry_type == Rectangle.geometry_name()
    assert rectangle.geometry == {"points": {"exterior": [[0, 0], [10, 10]], "interior": []}}
    assert rectangle.meta == {"note": "keep"}
    assert rectangle.tags == [{"tagId": 21, "value": None}]
    assert rectangle.priority == 1
    assert rectangle.custom_data == {"source": "model"}

    polygon = figures[201][0]
    assert polygon.class_id == 12
    assert polygon.geometry["points"]["exterior"] == [[1, 1], [5, 1], [5, 5]]


def test_alpha_geometries_are_stored_out_of_line(payload):
    figures, alpha_geometries = payload[4], payload[5]
    assert alpha_geometries == {ALPHA_FIGURE_ID: ALPHA_GEOMETRY}

    alpha_figure = next(f for f in figures[101] if f.id == ALPHA_FIGURE_ID)
    assert alpha_figure.geometry_type == AlphaMask.geometry_name()
    # The restore uploads alpha geometry separately, but a reader that asks for it gets
    # it attached rather than having to join the tables itself.
    assert alpha_figure.geometry == ALPHA_GEOMETRY


def test_class_names_are_stored_next_to_class_ids():
    """Ids belong to the source project; a comparison between snapshots matches on names."""
    import os
    import tempfile

    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.container import unpack_snapshot
    from supervisely.project.versioning.schema_fields import VersionSchemaField

    snapshot = Project.build_snapshot(
        _FakeApi(), project_id=PROJECT_ID, log_progress=False
    )
    with tempfile.TemporaryDirectory() as tmp:
        unpack_snapshot(snapshot.getvalue(), tmp)
        rows = image_snapshot_io.read_all_rows(
            tmp,
            image_snapshot_io.FIGURES_TABLE,
            columns=[VersionSchemaField.CLASS_ID, VersionSchemaField.CLASS_NAME],
        )
        assert {(r["class_id"], r["class_name"]) for r in rows} == {
            (11, "car"),
            (12, "road"),
            (13, "blob"),
        }
        # Reading two columns must not require the geometry column to exist in memory.
        assert set(rows[0]) == {"class_id", "class_name"}


def test_pickle_backups_still_load_through_the_same_entry_point():
    """v1.0.0 archives outlive the writer change; the sniff must keep choosing pickle."""
    source = (
        _project_info(),
        _meta(),
        list(_datasets()),
        [img for imgs in _images().values() for img in imgs],
        _figures(),
        {ALPHA_FIGURE_ID: ALPHA_GEOMETRY},
    )
    buffer = io.BytesIO()
    pickle.dump(source, buffer)
    buffer.seek(0)

    project_info, meta, dataset_infos, image_infos, figures, alpha_geometries = (
        Project._read_snapshot(buffer)
    )
    assert project_info.name == "cityscape"
    assert [c.name for c in meta.obj_classes] == ["car", "road", "blob"]
    assert len(dataset_infos) == 2
    assert len(image_infos) == 3
    assert set(figures) == {101, 201}
    assert alpha_geometries == {ALPHA_FIGURE_ID: ALPHA_GEOMETRY}


def test_repack_turns_a_pickle_backup_into_the_same_payload(payload):
    """Old versions reach the Parquet fast path by conversion, losing nothing on the way."""
    source = (
        _project_info(),
        _meta(),
        list(_datasets()),
        [img for imgs in _images().values() for img in imgs],
        _figures(),
        {ALPHA_FIGURE_ID: ALPHA_GEOMETRY},
    )
    buffer = io.BytesIO()
    pickle.dump(source, buffer)
    buffer.seek(0)

    repacked = Project._read_snapshot(Project.repack_snapshot(buffer))
    from_api = payload

    assert repacked[0] == from_api[0]
    assert repacked[1].to_json() == from_api[1].to_json()
    assert sorted(ds.id for ds in repacked[2]) == sorted(ds.id for ds in from_api[2])
    assert {img.id: img for img in repacked[3]} == {img.id: img for img in from_api[3]}
    assert {
        image_id: sorted(f.id for f in figs) for image_id, figs in repacked[4].items()
    } == {image_id: sorted(f.id for f in figs) for image_id, figs in from_api[4].items()}
    assert repacked[5] == from_api[5]


def test_repack_keeps_the_dataset_tree_paths():
    """A pickle payload has only parent_id, so the stored path has to be rebuilt from it."""
    import tempfile

    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.container import unpack_snapshot
    from supervisely.project.versioning.schema_fields import VersionSchemaField

    source = (
        _project_info(),
        _meta(),
        list(_datasets()),
        [],
        {},
        {},
    )
    buffer = io.BytesIO()
    pickle.dump(source, buffer)
    buffer.seek(0)

    with tempfile.TemporaryDirectory() as tmp:
        unpack_snapshot(Project.repack_snapshot(buffer).getvalue(), tmp)
        rows = image_snapshot_io.read_all_rows(tmp, image_snapshot_io.DATASETS_TABLE)
        paths = {
            r[VersionSchemaField.SRC_DATASET_ID]: r[VersionSchemaField.FULL_PATH] for r in rows
        }

    assert paths[1] == "train"
    assert paths[2] == os.path.join("train/datasets", "night")


# --------------------------------------------------------------------------- reader


def _write(tmp_path, snapshot) -> str:
    path = os.path.join(str(tmp_path), "version.bin")
    with open(path, "wb") as f:
        f.write(snapshot.getvalue())
    return path


def _pickle_snapshot() -> io.BytesIO:
    buffer = io.BytesIO()
    pickle.dump(
        (
            _project_info(),
            _meta(),
            list(_datasets()),
            [img for imgs in _images().values() for img in imgs],
            _figures(),
            {ALPHA_FIGURE_ID: ALPHA_GEOMETRY},
        ),
        buffer,
    )
    buffer.seek(0)
    return buffer


@pytest.fixture
def parquet_reader(tmp_path):
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    path = _write(tmp_path, Project.build_snapshot(_FakeApi(), PROJECT_ID, log_progress=False))
    with VersionSnapshot.open_archive(path) as snapshot:
        yield snapshot


@pytest.fixture
def pickle_reader(tmp_path):
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    path = _write(tmp_path, _pickle_snapshot())
    with VersionSnapshot.open_archive(path) as snapshot:
        yield snapshot


def test_reader_reports_the_format_it_got(parquet_reader, pickle_reader):
    """A caller budgeting memory has to be able to tell the two apart."""
    assert parquet_reader.is_columnar is True
    assert parquet_reader.schema_version == IMAGE_SCHEMA_VERSION_V2_1
    assert pickle_reader.is_columnar is False
    assert pickle_reader.project_type == "images"


def test_reader_exposes_meta_and_dataset_tree(parquet_reader):
    assert [c.name for c in parquet_reader.meta.obj_classes] == ["car", "road", "blob"]
    datasets = {ds.id: ds for ds in parquet_reader.datasets()}
    assert datasets[1].parent_id is None
    assert datasets[2].parent_id == 1
    assert datasets[2].full_path == os.path.join("train/datasets", "night")
    assert datasets[2].custom_data == {"weather": "rain"}


def _collect(batches):
    return [row for batch in batches for row in batch]


def test_both_backends_yield_the_same_canonical_rows(parquet_reader, pickle_reader):
    """The point of the reader: one shape, whatever the snapshot happens to store."""
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    def items(reader):
        return sorted(_collect(reader.iter_items()), key=lambda r: r[SnapshotColumn.ITEM_ID])

    def figures(reader):
        return sorted(
            _collect(reader.iter_figures()), key=lambda r: str(r[SnapshotColumn.FIGURE_ID])
        )

    assert items(parquet_reader) == items(pickle_reader)
    assert figures(parquet_reader) == figures(pickle_reader)

    first = items(parquet_reader)[0]
    assert first[SnapshotColumn.NAME] == "a.jpg"
    assert first[SnapshotColumn.HASH] == "hash-a"
    assert first[SnapshotColumn.WIDTH] == 800
    assert first[SnapshotColumn.FRAMES_COUNT] is None

    # Tags are their own stream now, and both backends agree there too.
    def tags(reader):
        return sorted(
            _collect(reader.iter_tags()), key=lambda r: (r[SnapshotColumn.OWNER_TYPE],
                                                         r[SnapshotColumn.OWNER_ID])
        )

    assert tags(parquet_reader) == tags(pickle_reader)
    assert {t[SnapshotColumn.OWNER_TYPE] for t in tags(parquet_reader)} == {"item", "figure"}

    # And they agree about reading one item's tags, which is all a comparison of two
    # versions ever asks for. The pickle used to take the argument and ignore it.
    def tags_of(reader, item_ids):
        return sorted(
            _collect(reader.iter_tags(item_ids=item_ids)),
            key=lambda r: (r[SnapshotColumn.OWNER_TYPE], r[SnapshotColumn.OWNER_ID]),
        )

    one = {row[SnapshotColumn.ITEM_ID] for row in tags(parquet_reader)}.pop()
    assert tags_of(parquet_reader, {one}) == tags_of(pickle_reader, {one})
    assert tags_of(pickle_reader, {one})
    assert all(row[SnapshotColumn.ITEM_ID] == one for row in tags_of(pickle_reader, {one}))
    assert tags_of(pickle_reader, {-1}) == []


def test_reader_projects_columns(parquet_reader):
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    wanted = [SnapshotColumn.ITEM_ID, SnapshotColumn.NAME]
    rows = _collect(parquet_reader.iter_items(columns=wanted))
    assert all(set(row) == set(wanted) for row in rows)
    assert sorted(row[SnapshotColumn.ITEM_ID] for row in rows) == [101, 102, 201]


def test_reader_batches_rows(parquet_reader):
    batches = list(parquet_reader.iter_items(batch_size=2))
    assert [len(batch) for batch in batches] == [2, 1]


def test_geometry_can_be_left_unread(parquet_reader):
    """Alpha payloads are most of a snapshot's bytes; counting figures must not pay for them."""
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    rows = _collect(parquet_reader.iter_figures(with_geometry=False))
    assert all(row[SnapshotColumn.GEOMETRY] is None for row in rows)
    assert {row[SnapshotColumn.CLASS_NAME] for row in rows} == {"car", "road", "blob"}

    with_geometry = {
        row[SnapshotColumn.FIGURE_ID]: row[SnapshotColumn.GEOMETRY]
        for row in _collect(parquet_reader.iter_figures())
    }
    assert with_geometry[ALPHA_FIGURE_ID] == ALPHA_GEOMETRY
    assert with_geometry[9001]["points"]["exterior"] == [[0, 0], [10, 10]]


def test_open_reuses_a_cached_snapshot(tmp_path):
    """A chain of comparisons must not download the same version once per pair."""
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    downloads = []

    class _VersionApi:
        def download_snapshot(self, project, version_id, dest_path=None):
            downloads.append(version_id)
            with open(dest_path, "wb") as f:
                f.write(
                    Project.build_snapshot(_FakeApi(), PROJECT_ID, log_progress=False).getvalue()
                )
            return dest_path

    class _Api:
        def __init__(self):
            self.project = type("_P", (), {"version": _VersionApi()})()

    api = _Api()
    cache_dir = os.path.join(str(tmp_path), "cache")
    for _ in range(2):
        with VersionSnapshot.open(api, PROJECT_ID, version_id=42, cache_dir=cache_dir) as snap:
            assert len(snap.datasets()) == 2
    assert downloads == [42]


def test_temporary_snapshots_are_cleaned_up(tmp_path):
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    path = _write(tmp_path, Project.build_snapshot(_FakeApi(), PROJECT_ID, log_progress=False))
    snapshot = VersionSnapshot.open_archive(path)
    payload_dir = snapshot._backend._payload_dir
    assert os.path.isdir(payload_dir)
    snapshot.close()
    assert not os.path.isdir(payload_dir)


def test_open_converts_a_legacy_snapshot_once_into_the_cache(tmp_path):
    """A pickle version stays slow on every read until it is converted; do it once."""
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn, VersionSnapshot

    downloads = []

    class _VersionApi:
        def download_snapshot(self, project, version_id, dest_path=None):
            downloads.append(version_id)
            with open(dest_path, "wb") as f:
                f.write(_pickle_snapshot().getvalue())
            return dest_path

    class _Api:
        def __init__(self):
            self.project = type("_P", (), {"version": _VersionApi()})()

    api = _Api()
    cache_dir = os.path.join(str(tmp_path), "cache")

    with VersionSnapshot.open(
        api, PROJECT_ID, version_id=7, cache_dir=cache_dir, repack_legacy=True
    ) as snapshot:
        assert snapshot.is_columnar is True
        first = sorted(row[SnapshotColumn.ITEM_ID] for row in _collect(snapshot.iter_items()))

    # Second open reads the converted copy: no download, and still columnar.
    with VersionSnapshot.open(api, PROJECT_ID, version_id=7, cache_dir=cache_dir) as snapshot:
        assert snapshot.is_columnar is True
        again = sorted(row[SnapshotColumn.ITEM_ID] for row in _collect(snapshot.iter_items()))
        assert again == first

    assert downloads == [7]


def test_open_without_repack_reads_a_legacy_snapshot_as_is(tmp_path):
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    class _VersionApi:
        def download_snapshot(self, project, version_id, dest_path=None):
            with open(dest_path, "wb") as f:
                f.write(_pickle_snapshot().getvalue())
            return dest_path

    class _Api:
        def __init__(self):
            self.project = type("_P", (), {"version": _VersionApi()})()

    with VersionSnapshot.open(
        _Api(), PROJECT_ID, version_id=7, cache_dir=os.path.join(str(tmp_path), "cache")
    ) as snapshot:
        assert snapshot.is_columnar is False
        assert len(_collect(snapshot.iter_items())) == 3


def test_new_image_snapshots_are_written_in_the_parquet_container_when_requested():
    """Data Versioning requests the current format explicitly; the public default stays v1."""
    from supervisely.project.versioning.common import DEFAULT_IMAGE_SCHEMA_VERSION
    from supervisely.project.versioning.container import is_snapshot_container, is_tar_container

    assert DEFAULT_IMAGE_SCHEMA_VERSION == IMAGE_SCHEMA_VERSION_V2_1

    snapshot = Project.download_bin(
        _FakeApi(),
        PROJECT_ID,
        return_bytesio=True,
        log_progress=False,
        schema_version=DEFAULT_IMAGE_SCHEMA_VERSION,
    )
    # A plain tar of already-compressed Parquet, not a second compression pass over it.
    assert is_tar_container(snapshot.getvalue())
    assert is_snapshot_container(snapshot.getvalue())


def test_the_legacy_writer_is_still_reachable_by_name():
    """v1.0.0 stays selectable so a caller pinned to the old format is not broken."""
    from supervisely.project.versioning.common import IMAGE_SCHEMA_VERSION_V1
    from supervisely.project.versioning.container import is_snapshot_container

    snapshot = Project.download_bin(
        _FakeApi(),
        PROJECT_ID,
        return_bytesio=True,
        log_progress=False,
        schema_version=IMAGE_SCHEMA_VERSION_V1,
    )
    assert not is_snapshot_container(snapshot.getvalue())
    assert Project._read_snapshot(snapshot)[0].name == "cityscape"


def test_the_public_download_default_stays_legacy():
    """A plain SDK install has no pyarrow, so an unchanged public call must stay usable."""
    from supervisely.project.versioning.container import is_snapshot_container

    snapshot = Project.download_bin(
        _FakeApi(), PROJECT_ID, return_bytesio=True, log_progress=False
    )

    assert not is_snapshot_container(snapshot.getvalue())


def test_data_version_explicitly_requests_the_current_image_format(monkeypatch):
    """Changing the public default back to v1 must not change newly created versions."""
    from supervisely.project.data_version import DataVersion
    from supervisely.project.versioning.common import DEFAULT_IMAGE_SCHEMA_VERSION

    requested = []

    def download_bin(*args, **kwargs):
        requested.append(kwargs.get("schema_version"))
        return io.BytesIO(b"snapshot")

    monkeypatch.setattr(Project, "download_bin", download_bin)
    api = SimpleNamespace(
        file=SimpleNamespace(upload=lambda *args, **kwargs: SimpleNamespace(id=1))
    )
    versions = DataVersion(api)
    versions.project_info = SimpleNamespace(id=PROJECT_ID, team_id=1, type="images")

    versions._compress_and_upload("/versions/test")

    assert requested == [DEFAULT_IMAGE_SCHEMA_VERSION]


def test_download_bin_forwards_the_requested_parquet_schema(tmp_path):
    """Requesting v2.0 must not silently stamp the current v2.1 schema."""
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    snapshot = Project.download_bin(
        _FakeApi(),
        PROJECT_ID,
        return_bytesio=True,
        log_progress=False,
        schema_version=IMAGE_SCHEMA_VERSION_V2,
    )
    path = _write(tmp_path, snapshot)

    with VersionSnapshot.open_archive(path) as reader:
        assert reader.schema_version == IMAGE_SCHEMA_VERSION_V2


def test_an_unknown_schema_version_is_refused():
    with pytest.raises(RuntimeError, match="schema_version"):
        Project.download_bin(_FakeApi(), PROJECT_ID, return_bytesio=True, schema_version="v9.9.9")


def test_repack_writes_alpha_rows_in_figure_order(tmp_path):
    """The reader walks the figures and alpha tables together instead of preloading the
    masks, which only works while the alpha rows follow the order of their figures."""
    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.container import unpack_snapshot
    from supervisely.project.versioning.schema_fields import VersionSchemaField

    second_alpha_id = 9004
    figures = _figures()
    figures[201].append(
        _make(
            FigureInfo,
            id=second_alpha_id,
            entity_id=201,
            dataset_id=2,
            class_id=13,
            geometry_type=AlphaMask.geometry_name(),
            geometry=None,
            geometry_meta={},
            meta={},
            tags=[],
        )
    )
    # Deliberately the reverse of the figure order, which is what a payload's own dict
    # order is under no obligation to match.
    alpha = {second_alpha_id: {"bitmap": {"data": "second"}}, ALPHA_FIGURE_ID: ALPHA_GEOMETRY}

    buffer = io.BytesIO()
    pickle.dump(
        (_project_info(), _meta(), list(_datasets()),
         [img for imgs in _images().values() for img in imgs], figures, alpha),
        buffer,
    )
    buffer.seek(0)

    repacked = Project.repack_snapshot(buffer)
    with tempfile.TemporaryDirectory() as tmp:
        unpack_snapshot(repacked.getvalue(), tmp)
        figure_ids = [
            r[VersionSchemaField.SRC_FIGURE_ID]
            for r in image_snapshot_io.read_all_rows(tmp, image_snapshot_io.FIGURES_TABLE)
        ]
        alpha_ids = [
            r[VersionSchemaField.SRC_FIGURE_ID]
            for r in image_snapshot_io.read_all_rows(tmp, image_snapshot_io.ALPHA_GEOMETRIES_TABLE)
        ]

    assert alpha_ids == [fid for fid in figure_ids if fid in alpha]

    # And the geometries still land on the right figures.
    repacked.seek(0)
    restored_figures = Project._read_snapshot(repacked)[4]
    by_id = {f.id: f for figs in restored_figures.values() for f in figs}
    assert by_id[ALPHA_FIGURE_ID].geometry == ALPHA_GEOMETRY
    assert by_id[second_alpha_id].geometry == {"bitmap": {"data": "second"}}


def test_projecting_columns_without_geometry_skips_the_alpha_table(parquet_reader):
    """with_geometry defaults to True, so a projected read must not silently pay for masks."""
    from supervisely.project.versioning import snapshot_reader
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    opened = []
    original = snapshot_reader._AlphaGeometryCursor

    class _Tracking(original):
        def __init__(self, *args, **kwargs):
            opened.append(1)
            super().__init__(*args, **kwargs)

    snapshot_reader._AlphaGeometryCursor = _Tracking
    try:
        rows = _collect(
            parquet_reader.iter_figures(
                columns=[SnapshotColumn.FIGURE_ID, SnapshotColumn.CLASS_NAME]
            )
        )
        assert not opened, "alpha table was walked for a read that discards geometry"
        assert len(rows) == 3

        with_geom = _collect(
            parquet_reader.iter_figures(
                columns=[SnapshotColumn.FIGURE_ID, SnapshotColumn.GEOMETRY]
            )
        )
        assert opened, "asking for geometry must still resolve alpha masks"
        assert {r[SnapshotColumn.FIGURE_ID]: r[SnapshotColumn.GEOMETRY] for r in with_geom}[
            ALPHA_FIGURE_ID
        ] == ALPHA_GEOMETRY
    finally:
        snapshot_reader._AlphaGeometryCursor = original


def test_item_filter_works_when_item_id_is_not_projected(parquet_reader):
    """The filter key must still be read even when the caller does not return it."""
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    rows = _collect(
        parquet_reader.iter_figures(
            columns=[SnapshotColumn.FIGURE_ID],
            with_geometry=False,
            item_ids={201},
        )
    )

    assert {row[SnapshotColumn.FIGURE_ID] for row in rows} == {9002}


def test_zstd_wrapped_snapshots_still_read(tmp_path):
    """Archives written before the codec change are tar.zst; they outlive the change."""
    import tarfile

    import zstd

    from supervisely.project.versioning.container import is_zstd_container, unpack_snapshot
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    plain = Project.build_snapshot(_FakeApi(), PROJECT_ID, log_progress=False).getvalue()

    # Re-wrap it the old way, byte for byte the shape the previous writer produced.
    payload = os.path.join(str(tmp_path), "payload")
    os.makedirs(payload)
    with tarfile.open(fileobj=io.BytesIO(plain), mode="r") as tar:
        tar.extractall(payload)
    tar_path = os.path.join(str(tmp_path), "old.tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(payload, arcname=".")
    legacy = os.path.join(str(tmp_path), "legacy.bin")
    with open(legacy, "wb") as f:
        f.write(zstd.compress(open(tar_path, "rb").read()))

    assert is_zstd_container(open(legacy, "rb").read(4))

    with VersionSnapshot.open_archive(legacy) as snapshot:
        assert snapshot.is_columnar is True
        assert len(_collect(snapshot.iter_items())) == 3

    with open(legacy, "rb") as f:
        payload_tuple = Project._read_snapshot(io.BytesIO(f.read()))
    assert payload_tuple[0].name == "cityscape"


def test_arrow_batches_carry_canonical_column_names(parquet_reader):
    """The fast path for a consumer that compares columns instead of rows."""
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    batches = list(
        parquet_reader.iter_figures_arrow(
            columns=[SnapshotColumn.FIGURE_ID, SnapshotColumn.CLASS_NAME]
        )
    )
    assert batches
    assert batches[0].schema.names == [SnapshotColumn.CLASS_NAME, SnapshotColumn.FIGURE_ID] or set(
        batches[0].schema.names
    ) == {SnapshotColumn.FIGURE_ID, SnapshotColumn.CLASS_NAME}
    assert sum(b.num_rows for b in batches) == 3

    items = list(parquet_reader.iter_items_arrow(columns=[SnapshotColumn.ITEM_ID]))
    assert sorted(items[0].column(0).to_pylist()) == [101, 102, 201]


def test_arrow_refuses_a_derived_column_instead_of_returning_nulls(parquet_reader):
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    with pytest.raises(ValueError, match="derived"):
        list(parquet_reader.iter_items_arrow(columns=[SnapshotColumn.FRAMES_COUNT]))


def test_arrow_is_not_offered_for_the_pickle_format(pickle_reader):
    """A pickle has no columns to hand out; saying so beats a slow fake."""
    with pytest.raises(NotImplementedError, match="not stored as columns"):
        list(pickle_reader.iter_items_arrow())


def test_figure_timestamps_survive_and_are_readable(parquet_reader):
    """The pickle format carried them on every FigureInfo; a comparison between two
    versions of one project can use them to skip work, so the new format keeps them."""
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    rows = {
        r[SnapshotColumn.FIGURE_ID]: r
        for r in _collect(
            parquet_reader.iter_figures(
                columns=[
                    SnapshotColumn.FIGURE_ID,
                    SnapshotColumn.CREATED_AT,
                    SnapshotColumn.UPDATED_AT,
                ]
            )
        )
    }
    assert rows[9001][SnapshotColumn.CREATED_AT] == "2026-03-01T00:00:00.000Z"
    assert rows[9001][SnapshotColumn.UPDATED_AT] == "2026-03-02T00:00:00.000Z"

    # And through the restore payload, which is what upload_bin works from.
    payload_figures = Project._read_snapshot(
        Project.build_snapshot(_FakeApi(), PROJECT_ID, log_progress=False)
    )[4]
    figure = next(f for figs in payload_figures.values() for f in figs if f.id == 9001)
    assert figure.created_at == "2026-03-01T00:00:00.000Z"
    assert figure.updated_at == "2026-03-02T00:00:00.000Z"


def test_image_snapshots_are_numbered_like_video_and_volume_ones():
    """One version string, one meaning across modalities - that is what the diff gate reads."""
    from supervisely.project.versioning.common import (
        DEFAULT_IMAGE_SCHEMA_VERSION,
        DEFAULT_VIDEO_SCHEMA_VERSION,
        DEFAULT_VOLUME_SCHEMA_VERSION,
    )

    assert DEFAULT_IMAGE_SCHEMA_VERSION == DEFAULT_VIDEO_SCHEMA_VERSION
    assert DEFAULT_IMAGE_SCHEMA_VERSION == DEFAULT_VOLUME_SCHEMA_VERSION
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    assert DEFAULT_IMAGE_SCHEMA_VERSION == VersionSnapshot.DIFFABLE_SCHEMA_VERSION


def test_a_new_snapshot_may_be_diffed_and_a_pickle_may_not(parquet_reader, pickle_reader):
    """The strict rule: comparison is offered only on the current format."""
    assert parquet_reader.is_diffable is True
    assert parquet_reader.diff_unsupported_reason is None

    from supervisely.project.versioning.common import IMAGE_SCHEMA_VERSION_V1

    assert pickle_reader.is_diffable is False
    assert IMAGE_SCHEMA_VERSION_V1 in pickle_reader.diff_unsupported_reason


def test_a_version_can_be_judged_from_its_recorded_format_alone():
    """What the version list is gated on. `versions.json` records the format of every
    version, so which of them are worth offering for comparison is answered without
    downloading one - a version written before the format was recorded has none."""
    from supervisely.project.versioning.common import (
        DEFAULT_IMAGE_SCHEMA_VERSION,
        IMAGE_SCHEMA_VERSION_V1,
        IMAGE_SCHEMA_VERSION_V2,
    )
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    assert VersionSnapshot.format_is_diffable(DEFAULT_IMAGE_SCHEMA_VERSION) is True
    assert VersionSnapshot.format_is_diffable(IMAGE_SCHEMA_VERSION_V2) is False
    assert VersionSnapshot.format_is_diffable(IMAGE_SCHEMA_VERSION_V1) is False
    assert VersionSnapshot.format_is_diffable(None) is False
    assert VersionSnapshot.format_is_diffable("") is False


def test_snapshots_written_before_the_renumbering_still_read(tmp_path):
    """v2.0.0 and v2.1.0 are the same layout, so the older number stays readable."""
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    path = _write(
        tmp_path,
        Project.build_snapshot(
            _FakeApi(),
            PROJECT_ID,
            log_progress=False,
            schema_version=IMAGE_SCHEMA_VERSION_V2,
        ),
    )
    with VersionSnapshot.open_archive(path) as snap:
        assert snap.schema_version == IMAGE_SCHEMA_VERSION_V2
        assert snap.is_diffable is False
        assert [row["name"] for batch in snap.iter_items() for row in batch]


def test_an_alpha_mask_survives_a_read_that_does_not_ask_for_the_figure_id(parquet_reader):
    """The mask lives in a table of its own, keyed by figure id.

    A caller that wants geometry and nothing else never mentions that id, and the read used
    to project it away - so the lookup was made with `None` and every alpha mask came back
    empty, with no error anywhere to say so.
    """
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    rows = _collect(
        parquet_reader.iter_figures(
            columns=[SnapshotColumn.GEOMETRY_TYPE, SnapshotColumn.GEOMETRY]
        )
    )
    masks = [
        row
        for row in rows
        if row[SnapshotColumn.GEOMETRY_TYPE] == AlphaMask.geometry_name()
    ]

    assert len(masks) == 1
    assert masks[0][SnapshotColumn.GEOMETRY] == ALPHA_GEOMETRY
    # Asked for two columns, given two: the id is read to find the mask, not handed back.
    assert set(masks[0]) == {SnapshotColumn.GEOMETRY_TYPE, SnapshotColumn.GEOMETRY}


def test_the_alpha_cursor_stops_hoarding_rows_it_walks_past(tmp_path):
    """Both tables are written in figure order, so walking them together normally keeps
    nothing. A reader filtering by item skips most figures, and every mask in between was
    kept forever - the whole mask table, for a diff that touches a handful of items."""
    import json

    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.schema_fields import VersionSchemaField
    from supervisely.project.versioning.snapshot_reader import _AlphaGeometryCursor

    rows = [
        {
            VersionSchemaField.SRC_FIGURE_ID: figure_id,
            VersionSchemaField.GEOMETRY_JSON: json.dumps({"id": figure_id}),
        }
        for figure_id in range(10_000)
    ]

    def _batches(payload_dir, table, batch_size=5000, columns=None):
        for start in range(0, len(rows), 500):
            yield rows[start : start + 500]

    original = image_snapshot_io.iter_rows
    image_snapshot_io.iter_rows = _batches
    try:
        cursor = _AlphaGeometryCursor("unused", 500)
        # The one figure this reader cares about is at the far end of the table.
        assert cursor.get(9_999) == {"id": 9_999}
        # Against a fixed number rather than the limit itself, so raising the limit does not
        # quietly make this pass again: unbounded, it would be holding all 9_999 it walked.
        assert len(cursor._stash) < 5_000
        # Having given up on the merge, it still answers - by reading the table again.
        assert cursor.get(3) == {"id": 3}
    finally:
        image_snapshot_io.iter_rows = original


def test_a_figure_row_carries_object_id_even_though_images_have_none(parquet_reader, pickle_reader):
    """`FIGURE_COLUMNS` promises the key for every modality.

    An image figure is the label, so there is no object above it and the value is null - but
    a reader written against the canonical columns and handed `columns=None` used to get a
    dict without the key at all, and a KeyError instead of a None.
    """
    from supervisely.project.versioning.snapshot_reader import FIGURE_COLUMNS, SnapshotColumn

    for reader in (parquet_reader, pickle_reader):
        rows = _collect(reader.iter_figures())

        assert rows
        for row in rows:
            assert SnapshotColumn.OBJECT_ID in row
            assert row[SnapshotColumn.OBJECT_ID] is None
        # Nothing else the canonical list promises is missing either.
        assert set(FIGURE_COLUMNS) <= set(rows[0])


def test_a_projection_that_maps_to_nothing_reads_nothing(parquet_reader):
    """`OBJECT_ID` is the one canonical column an image snapshot has no store for.

    Asking for it alone used to leave the physical projection empty, and empty meant "every
    column" one level down - so the narrowest possible read became the widest, geometry and
    all.
    """
    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    asked = []
    original = image_snapshot_io.iter_rows

    def _recording(payload_dir, table, columns=None, batch_size=5000):
        asked.append((table, None if columns is None else sorted(columns)))
        return original(payload_dir, table, columns=columns, batch_size=batch_size)

    image_snapshot_io.iter_rows = _recording
    try:
        rows = _collect(
            parquet_reader.iter_figures(columns=[SnapshotColumn.OBJECT_ID], with_geometry=False)
        )
    finally:
        image_snapshot_io.iter_rows = original

    assert [row[SnapshotColumn.OBJECT_ID] for row in rows] == [None, None, None]
    figures_reads = [columns for table, columns in asked if table == image_snapshot_io.FIGURES_TABLE]
    assert figures_reads == [[]], f"read more than asked for: {figures_reads}"


def test_geometry_is_not_read_when_it_is_not_wanted(parquet_reader):
    """The heaviest column in the table, and the whole point of `with_geometry=False`.

    It only ever lands in the projection when the caller names no columns at all, since
    that asks for every canonical one - so that is the read this is about. The video
    backend already dropped it there; images read it and threw it away.
    """
    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.schema_fields import VersionSchemaField
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    asked = []
    original = image_snapshot_io.iter_rows

    def _recording(payload_dir, table, columns=None, batch_size=5000):
        if table == image_snapshot_io.FIGURES_TABLE:
            asked.append(columns)
        return original(payload_dir, table, columns=columns, batch_size=batch_size)

    image_snapshot_io.iter_rows = _recording
    try:
        rows = _collect(parquet_reader.iter_figures(with_geometry=False))
    finally:
        image_snapshot_io.iter_rows = original

    assert asked and all(
        VersionSchemaField.GEOMETRY_JSON not in (columns or []) for columns in asked
    ), f"geometry was read and discarded: {asked}"
    # Every other canonical column still comes back; only the payload is left behind.
    assert rows and all(row[SnapshotColumn.GEOMETRY] is None for row in rows)
    assert rows[0][SnapshotColumn.CLASS_NAME] is not None


def test_an_interrupted_extraction_is_not_mistaken_for_a_finished_one(tmp_path):
    """The manifest lands early in the archive.

    Keying reuse on it meant an extraction killed part-way left a payload that every later
    open took for complete - and then read a table that was not there.
    """
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    path = _write(tmp_path, Project.build_snapshot(_FakeApi(), PROJECT_ID, log_progress=False))
    payload_dir = str(tmp_path / "payload")

    os.makedirs(payload_dir, exist_ok=True)
    # What an extraction that died right after the manifest leaves behind.
    with open(os.path.join(payload_dir, "manifest.json"), "w", encoding="utf-8") as f:
        f.write('{"schemaVersion": "v2.1.0"}')

    with VersionSnapshot.open_archive(path, payload_dir=payload_dir) as snapshot:
        assert _collect(snapshot.iter_items()), "the payload was not extracted again"


def test_a_failed_figure_tag_read_fails_the_snapshot():
    """No quiet fallback to the lean figures.list tags.

    Those carry no name and no updated_at, and a version is immutable: one written that
    way would report every figure tag on a touched item as changed against its neighbours.
    """
    api = _FakeApi()

    def broken(dataset_id, image_ids):
        raise RuntimeError("annotations.bulk.info: 502")

    api.annotation._figure_tags_batch = broken
    with pytest.raises(RuntimeError, match="502"):
        Project.build_snapshot(api, PROJECT_ID, log_progress=False)


def test_a_filtered_geometry_read_walks_the_mask_table_once(parquet_reader, monkeypatch):
    """With an item filter the masks are looked up by id, not merged: once past the cursor's
    stash limit every wanted mask used to cost a scan of the whole alpha table."""
    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.snapshot_reader import SnapshotColumn

    alpha_reads = []
    original = image_snapshot_io.iter_rows

    def counting(payload_dir, table, *args, **kwargs):
        if table == image_snapshot_io.ALPHA_GEOMETRIES_TABLE:
            alpha_reads.append(table)
        return original(payload_dir, table, *args, **kwargs)

    monkeypatch.setattr(image_snapshot_io, "iter_rows", counting)

    rows = _collect(parquet_reader.iter_figures(item_ids={101}))
    by_id = {row[SnapshotColumn.FIGURE_ID]: row for row in rows}
    assert set(by_id) == {9001, ALPHA_FIGURE_ID}
    assert by_id[ALPHA_FIGURE_ID][SnapshotColumn.GEOMETRY] == ALPHA_GEOMETRY
    assert len(alpha_reads) == 1

    # An item with no masks never opens the mask table at all.
    alpha_reads.clear()
    assert [row[SnapshotColumn.FIGURE_ID] for row in _collect(
        parquet_reader.iter_figures(item_ids={201})
    )] == [9002]
    assert alpha_reads == []


def test_a_repacked_pickle_does_not_pass_as_comparable(tmp_path):
    """A pickle kept figure tags as figures.list rendered them - no name, no updated_at - so
    a snapshot converted from one must not claim the format a comparison accepts."""
    from supervisely.project.versioning.snapshot_reader import VersionSnapshot

    path = _write(tmp_path, Project.repack_snapshot(_pickle_snapshot()))
    with VersionSnapshot.open_archive(path) as snap:
        assert snap.schema_version == IMAGE_SCHEMA_VERSION_V2
        assert snap.is_diffable is False


def test_a_filtered_cursor_drops_the_masks_it_does_not_want(monkeypatch):
    """Given the wanted ids, rows passed over are dropped rather than stashed, so a filtered
    read holds no mask it was not asked for."""
    import json

    from supervisely.project.versioning import image_snapshot_io
    from supervisely.project.versioning.schema_fields import VersionSchemaField
    from supervisely.project.versioning.snapshot_reader import _AlphaGeometryCursor

    rows = [
        {
            VersionSchemaField.SRC_FIGURE_ID: figure_id,
            VersionSchemaField.GEOMETRY_JSON: json.dumps({"id": figure_id}),
        }
        for figure_id in range(10_000)
    ]
    monkeypatch.setattr(
        image_snapshot_io,
        "iter_rows",
        lambda *a, **k: (rows[i : i + 500] for i in range(0, len(rows), 500)),
    )
    cursor = _AlphaGeometryCursor("unused", 500, wanted={17, 9_999})
    assert cursor.get(9_999) == {"id": 9_999}
    # 17 was passed before 9_999 was reached, and is the one row worth keeping.
    assert set(cursor._stash) == {17}
    assert cursor.get(17) == {"id": 17}


def test_figure_tags_come_from_the_annotation_endpoint_and_skip_deleted_images():
    """annotations.bulk.info renders a figure tag with its name and timestamps. An image
    deleted while the snapshot ran is simply not returned - it has no tags to give, and
    download_batch's ordering would have raised on it."""
    from supervisely.api.annotation_api import AnnotationApi

    posted = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def post(method, data):
        posted.append((method, list(data["imageIds"])))
        return _Response(
            [
                {
                    "imageId": 101,
                    "annotation": {
                        "objects": [
                            {"id": 9001, "tags": [{"id": 5, "tagId": 21, "name": "score"}]},
                            {"id": 9003, "tags": []},
                        ]
                    },
                }
            ]
        )

    api = AnnotationApi.__new__(AnnotationApi)
    api._api = SimpleNamespace(post=post)

    tags = api._figure_tags_batch(1, [101, 102])

    assert tags == {9001: [{"id": 5, "tagId": 21, "name": "score"}]}
    assert posted == [("annotations.bulk.info", [101, 102])]


def test_the_fetch_pool_keeps_a_window_in_flight_not_the_page(monkeypatch):
    """pool.map submits every batch of a page at once and holds each finished one - masks
    and all - until the writer reaches it."""
    import threading

    from supervisely.project import project as project_module

    in_flight = []
    live = [0]
    lock = threading.Lock()
    api = _FakeApi()
    original = api.image.figure.download

    def slow_download(dataset_id, image_ids, integer_coords=False):
        with lock:
            live[0] += 1
            in_flight.append(live[0])
        try:
            return original(dataset_id, image_ids, integer_coords=integer_coords)
        finally:
            with lock:
                live[0] -= 1

    api.image.figure.download = slow_download
    Project.build_snapshot(api, PROJECT_ID, batch_size=1, log_progress=False, fetch_workers=2)
    assert max(in_flight) <= 2
