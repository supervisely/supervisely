# coding: utf-8
"""Round-trip tests for the Parquet image snapshot (schema v2.0.0).

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
from supervisely.project.versioning.common import IMAGE_SCHEMA_VERSION_V2

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


class _FakeImageApi:
    def __init__(self, images, figures):
        self._images = images
        self.figure = _FakeFigureApi(figures)

    def get_list(self, dataset_id):
        return self._images.get(dataset_id, [])

    def get_list_generator(self, dataset_id, batch_size=None):
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
    assert parquet_reader.schema_version == IMAGE_SCHEMA_VERSION_V2
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


def test_new_image_snapshots_are_written_in_the_parquet_container():
    """The writer default. A revert to pickle would silently undo the whole change."""
    from supervisely.project.versioning.common import DEFAULT_IMAGE_SCHEMA_VERSION
    from supervisely.project.versioning.container import is_snapshot_container, is_tar_container

    assert DEFAULT_IMAGE_SCHEMA_VERSION == IMAGE_SCHEMA_VERSION_V2

    snapshot = Project.download_bin(_FakeApi(), PROJECT_ID, return_bytesio=True, log_progress=False)
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
