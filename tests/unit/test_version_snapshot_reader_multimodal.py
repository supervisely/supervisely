# coding: utf-8
"""The snapshot reader against video and volume snapshots.

Those two formats already existed, so what is under test here is the mapping onto the
reader's canonical columns - the places where a rename is easy to get wrong: a video's
``frame_width`` becoming ``width``, a figure's class name coming from the *objects*
table rather than the figure row, and a volume's figures having to be flattened out of
one annotation document per item.

The payloads are built here rather than downloaded, using the same schema objects the
writers use, so a change to those schemas fails this test instead of failing in a diff.
"""

import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow", reason="snapshots need the 'versioning' extra")

import pyarrow
import pyarrow.parquet as parquet

from supervisely.annotation.label import LabelJsonFields
from supervisely.annotation.obj_class import ObjClass
from supervisely.annotation.obj_class_collection import ObjClassCollection
from supervisely.api.module_api import ApiField
from supervisely.geometry.rectangle import Rectangle
from supervisely.io.json import dump_json_file
from supervisely.project.project_meta import ProjectMeta
from supervisely.project.versioning.common import (
    get_video_snapshot_schema,
    get_volume_snapshot_schema,
)
from supervisely.project.versioning.container import pack_payload_dir, table_meta
from supervisely.project.versioning.schema_fields import VersionSchemaField
from supervisely.project.versioning.snapshot_reader import SnapshotColumn, VersionSnapshot
from supervisely.video_annotation import constants as video_constants
from supervisely.volume_annotation import constants as volume_constants

META = ProjectMeta(obj_classes=ObjClassCollection([ObjClass("car", Rectangle, color=[1, 2, 3])]))
GEOMETRY = {"points": {"exterior": [[0, 0], [4, 4]], "interior": []}}


def _write_payload(tmp_path, project_type, tables):
    """Write a snapshot payload and pack it, returning the path to the archive."""
    payload_dir = os.path.join(str(tmp_path), "payload")
    os.makedirs(payload_dir, exist_ok=True)

    dump_json_file(
        {"id": 1, "name": "p", "type": project_type},
        os.path.join(payload_dir, "project_info.json"),
    )
    dump_json_file(META.to_json(), os.path.join(payload_dir, "project_meta.json"))

    tables_meta = []
    for name, (schema, rows) in tables.items():
        table = pyarrow.Table.from_pylist(rows, schema=schema)
        parquet.write_table(table, os.path.join(payload_dir, f"{name}.parquet"))
        tables_meta.append(table_meta(name, f"{name}.parquet", table.num_rows))

    dump_json_file(
        {
            VersionSchemaField.SCHEMA_VERSION: "v2.0.0",
            VersionSchemaField.TABLES: tables_meta,
        },
        os.path.join(payload_dir, "manifest.json"),
    )

    archive_path = os.path.join(str(tmp_path), "version.bin")
    with open(archive_path, "wb") as f:
        f.write(pack_payload_dir(payload_dir, str(tmp_path)).getvalue())
    return archive_path


def _parquet_bytes(table):
    buf = pyarrow.BufferOutputStream()
    parquet.write_table(table, buf)
    return buf.getvalue().to_pybytes()


def _collect(batches):
    return [row for batch in batches for row in batch]


@pytest.fixture
def video_snapshot(tmp_path):
    schema = get_video_snapshot_schema("v2.0.0")
    tables = {
        "datasets": (
            schema.datasets_schema(pyarrow),
            [
                {
                    VersionSchemaField.SRC_DATASET_ID: 1,
                    VersionSchemaField.PARENT_SRC_DATASET_ID: None,
                    VersionSchemaField.NAME: "clips",
                    VersionSchemaField.FULL_PATH: "clips",
                    VersionSchemaField.DESCRIPTION: None,
                    VersionSchemaField.CUSTOM_DATA: None,
                }
            ],
        ),
        "videos": (
            schema.videos_schema(pyarrow),
            [
                schema.video_row(
                    src_video_id=10,
                    src_dataset_id=1,
                    name="drive.mp4",
                    hash="hash-v",
                    link=None,
                    frames_count=120,
                    frame_width=1920,
                    frame_height=1080,
                    frames_to_timecodes=None,
                    meta={"fps": 30},
                    custom_data=None,
                    created_at="2026-01-01T00:00:00.000Z",
                    updated_at="2026-01-02T00:00:00.000Z",
                    ann_json={},
                )
            ],
        ),
        "objects": (
            schema.objects_schema(pyarrow),
            [
                {
                    VersionSchemaField.SRC_OBJECT_ID: 5,
                    VersionSchemaField.SRC_VIDEO_ID: 10,
                    VersionSchemaField.CLASS_NAME: "car",
                    VersionSchemaField.KEY: "abc",
                    VersionSchemaField.TAGS_JSON: json.dumps([{"name": "reviewed"}]),
                },
                # Tagged, never drawn: no figure points at it, so nothing but the objects
                # table knows it exists or what class it is.
                {
                    VersionSchemaField.SRC_OBJECT_ID: 6,
                    VersionSchemaField.SRC_VIDEO_ID: 10,
                    VersionSchemaField.CLASS_NAME: "truck",
                    VersionSchemaField.KEY: "def",
                    VersionSchemaField.TAGS_JSON: None,
                },
            ],
        ),
        "figures": (
            schema.figures_schema(pyarrow),
            [
                {
                    VersionSchemaField.SRC_FIGURE_ID: 100,
                    VersionSchemaField.SRC_OBJECT_ID: 5,
                    VersionSchemaField.SRC_VIDEO_ID: 10,
                    VersionSchemaField.FRAME_INDEX: 7,
                    VersionSchemaField.GEOMETRY_TYPE: Rectangle.geometry_name(),
                    VersionSchemaField.GEOMETRY_JSON: json.dumps(GEOMETRY),
                }
            ],
        ),
    }
    with VersionSnapshot.open_archive(_write_payload(tmp_path, "videos", tables)) as snapshot:
        yield snapshot


def test_video_items_map_frame_size_onto_the_canonical_columns(video_snapshot):
    assert video_snapshot.project_type == "videos"
    assert video_snapshot.is_columnar is True

    item = _collect(video_snapshot.iter_items())[0]
    assert item[SnapshotColumn.ITEM_ID] == 10
    assert item[SnapshotColumn.DATASET_ID] == 1
    assert item[SnapshotColumn.NAME] == "drive.mp4"
    assert item[SnapshotColumn.HASH] == "hash-v"
    # The video schema calls these frame_width/frame_height.
    assert item[SnapshotColumn.WIDTH] == 1920
    assert item[SnapshotColumn.HEIGHT] == 1080
    assert item[SnapshotColumn.FRAMES_COUNT] == 120
    assert item[SnapshotColumn.META] == {"fps": 30}


def test_video_figures_take_class_and_tags_from_their_object(video_snapshot):
    figure = _collect(video_snapshot.iter_figures())[0]
    assert figure[SnapshotColumn.FIGURE_ID] == 100
    assert figure[SnapshotColumn.ITEM_ID] == 10
    assert figure[SnapshotColumn.FRAME_INDEX] == 7
    # class_name is not on the figure row; it comes from the objects table.
    assert figure[SnapshotColumn.CLASS_NAME] == "car"
    assert figure[SnapshotColumn.GEOMETRY] == GEOMETRY

    without = _collect(video_snapshot.iter_figures(with_geometry=False))[0]
    assert without[SnapshotColumn.GEOMETRY] is None


def test_figures_name_the_object_they_belong_to(video_snapshot):
    """A video figure is one frame of an object that spans frames, and a comparison that
    wants to report per object rather than per figure needs the link.

    The column was already being read to join the class name; surfacing it costs nothing.
    """
    figure = _collect(video_snapshot.iter_figures())[0]
    assert figure[SnapshotColumn.OBJECT_ID] == 5

    projected = _collect(
        video_snapshot.iter_figures(
            columns=[SnapshotColumn.FIGURE_ID, SnapshotColumn.OBJECT_ID], with_geometry=False
        )
    )
    assert projected[0] == {SnapshotColumn.FIGURE_ID: 100, SnapshotColumn.OBJECT_ID: 5}


def test_video_item_filter_works_when_item_id_is_not_projected(video_snapshot):
    """Projection must not remove the physical column used by item_ids filtering."""
    included = _collect(
        video_snapshot.iter_figures(
            columns=[SnapshotColumn.FIGURE_ID], with_geometry=False, item_ids={10}
        )
    )
    excluded = _collect(
        video_snapshot.iter_figures(
            columns=[SnapshotColumn.FIGURE_ID], with_geometry=False, item_ids={999}
        )
    )

    assert included == [{SnapshotColumn.FIGURE_ID: 100}]
    assert excluded == []


def test_objects_can_be_read_without_going_through_their_figures(video_snapshot):
    """The class of a video object lives in the objects table; a figure reports one only
    because iter_figures joins that table in. An object with no figures - tagged but never
    drawn - is therefore invisible to every other reader, and it is exactly the object a
    diff has nothing to name."""
    objects = _collect(video_snapshot.iter_objects())

    assert [obj[SnapshotColumn.OBJECT_ID] for obj in objects] == [5, 6]
    assert [obj[SnapshotColumn.CLASS_NAME] for obj in objects] == ["car", "truck"]
    assert objects[0][SnapshotColumn.ITEM_ID] == 10

    projected = _collect(
        video_snapshot.iter_objects(
            columns=[SnapshotColumn.OBJECT_ID, SnapshotColumn.CLASS_NAME]
        )
    )
    assert projected[1] == {SnapshotColumn.OBJECT_ID: 6, SnapshotColumn.CLASS_NAME: "truck"}


@pytest.fixture
def volume_snapshot(tmp_path):
    schema = get_volume_snapshot_schema("v2.0.0")
    annotation = {
        volume_constants.OBJECTS: [
            {
                volume_constants.KEY: "obj-1",
                LabelJsonFields.OBJ_CLASS_NAME: "car",
                volume_constants.TAGS: [{"name": "checked"}],
            }
        ],
        volume_constants.PLANES: [
            {
                volume_constants.NAME: "axial",
                volume_constants.SLICES: [
                    {
                        volume_constants.INDEX: 3,
                        volume_constants.FIGURES: [
                            {
                                volume_constants.KEY: "fig-1",
                                volume_constants.OBJECT_KEY: "obj-1",
                                ApiField.GEOMETRY_TYPE: Rectangle.geometry_name(),
                                ApiField.GEOMETRY: GEOMETRY,
                            }
                        ],
                    }
                ],
            }
        ],
        volume_constants.SPATIAL_FIGURES: [
            {
                volume_constants.KEY: "fig-2",
                volume_constants.OBJECT_KEY: "obj-1",
                ApiField.GEOMETRY_TYPE: "mask_3d",
                ApiField.GEOMETRY: {"mask3DId": "x"},
            }
        ],
    }
    # The real container: one blob with its own header holding the Parquet tables as
    # sections, not a tar of files. Built through VolumeProject's own assembler so a
    # change to that format fails here.
    from supervisely.project.volume_project import VolumeProject

    def parquet_bytes(table):
        buf = pyarrow.BufferOutputStream()
        parquet.write_table(table, buf)
        return buf.getvalue().to_pybytes()

    datasets = pyarrow.Table.from_pylist(
        [schema.dataset_row_from_record({ApiField.ID: 1, ApiField.NAME: "scans"})],
        schema=schema.datasets_table_schema(pyarrow),
    )
    volumes = pyarrow.Table.from_pylist(
        [
            schema.volume_row_from_record(
                {
                    ApiField.ID: 20,
                    ApiField.DATASET_ID: 1,
                    ApiField.NAME: "ct.nrrd",
                    ApiField.HASH: "hash-vol",
                }
            )
        ],
        schema=schema.volumes_table_schema(pyarrow),
    )
    anns = pyarrow.Table.from_pylist(
        [schema.annotation_row_from_dict(src_volume_id=20, annotation=annotation)],
        schema=schema.annotations_table_schema(pyarrow),
    )

    blob = VolumeProject._assemble_sections(
        [
            (
                VolumeProject._SECTION_PROJECT_INFO,
                json.dumps({"id": 1, "name": "p", "type": "volumes"}).encode(),
            ),
            (VolumeProject._SECTION_PROJECT_META, json.dumps(META.to_json()).encode()),
            (VolumeProject._SECTION_DATASETS, parquet_bytes(datasets)),
            (VolumeProject._SECTION_VOLUMES, parquet_bytes(volumes)),
            (VolumeProject._SECTION_ANNOTATIONS, parquet_bytes(anns)),
        ]
    )
    archive_path = os.path.join(str(tmp_path), "volume.bin")
    with open(archive_path, "wb") as f:
        f.write(blob)

    with VersionSnapshot.open_archive(archive_path) as snapshot:
        yield snapshot


def test_volume_items_come_out_of_the_stored_json_records(volume_snapshot):
    assert volume_snapshot.project_type == "volumes"

    datasets = volume_snapshot.datasets()
    assert [ds.name for ds in datasets] == ["scans"]

    item = _collect(volume_snapshot.iter_items())[0]
    assert item[SnapshotColumn.ITEM_ID] == 20
    assert item[SnapshotColumn.DATASET_ID] == 1
    assert item[SnapshotColumn.NAME] == "ct.nrrd"
    assert item[SnapshotColumn.HASH] == "hash-vol"


def test_volume_figures_are_flattened_out_of_the_annotation(volume_snapshot):
    figures = {
        row[SnapshotColumn.FIGURE_ID]: row for row in _collect(volume_snapshot.iter_figures())
    }
    assert set(figures) == {"fig-1", "fig-2"}

    sliced = figures["fig-1"]
    assert sliced[SnapshotColumn.ITEM_ID] == 20
    assert sliced[SnapshotColumn.FRAME_INDEX] == 3
    assert sliced[SnapshotColumn.CLASS_NAME] == "car"
    assert sliced[SnapshotColumn.GEOMETRY] == GEOMETRY

    # The tag on that object comes out of the tags stream, not off the figure.
    tags = _collect(volume_snapshot.iter_tags())
    assert [t[SnapshotColumn.NAME] for t in tags] == ["checked"]
    assert tags[0][SnapshotColumn.OWNER_TYPE] == "object"

    # A spatial figure belongs to the volume, not to a slice.
    spatial = figures["fig-2"]
    assert spatial[SnapshotColumn.FRAME_INDEX] is None
    assert spatial[SnapshotColumn.CLASS_NAME] == "car"

    # Volume objects have keys rather than ids, and that key is what the object column
    # carries - the same value the object's own tags are attributed to.
    assert sliced[SnapshotColumn.OBJECT_ID] == "obj-1"
    assert spatial[SnapshotColumn.OBJECT_ID] == "obj-1"
    assert tags[0][SnapshotColumn.OWNER_ID] == "obj-1"


# ------------------------------------------------------- the video writer itself


def _volume_path(tmp_path, name, annotation, src_volume_id=20):
    """The same blob, written where open_archive can read it."""
    path = os.path.join(str(tmp_path), name)
    with open(path, "wb") as f:
        f.write(_volume_blob(annotation, src_volume_id))
    return path


def _volume_blob(annotation, src_volume_id=20):
    """One volume snapshot blob holding a single annotation record."""
    from supervisely.project.volume_project import VolumeProject

    schema = get_volume_snapshot_schema("v2.0.0")

    return VolumeProject._assemble_sections(
        [
            (
                VolumeProject._SECTION_PROJECT_INFO,
                json.dumps({"id": 1, "name": "p", "type": "volumes"}).encode(),
            ),
            (VolumeProject._SECTION_PROJECT_META, json.dumps(META.to_json()).encode()),
            (
                VolumeProject._SECTION_ANNOTATIONS,
                _parquet_bytes(
                    pyarrow.Table.from_pylist(
                        [
                            schema.annotation_row_from_dict(
                                src_volume_id=src_volume_id, annotation=annotation
                            )
                        ],
                        schema=schema.annotations_table_schema(pyarrow),
                    )
                ),
            ),
        ]
    )


def test_a_volume_annotation_keeps_the_server_ids_of_its_slice_figures(tmp_path):
    """The identity two versions are compared on. `to_json` handed its key map to the tags,
    the objects and the spatial figures and not to the planes, so every slice figure went
    into the snapshot with nothing but the uuid key the SDK invents on each parse - and a
    diff over that reports every figure of a touched volume as removed and added again."""
    from supervisely.video_annotation.key_id_map import KeyIdMap
    from supervisely.volume_annotation.volume_annotation import VolumeAnnotation

    annotation = {
        volume_constants.OBJECTS: [
            {
                volume_constants.ID: 1265,
                LabelJsonFields.OBJ_CLASS_NAME: "car",
                volume_constants.TAGS: [],
            }
        ],
        volume_constants.PLANES: [
            {
                volume_constants.NAME: name,
                volume_constants.NORMAL: normal,
                volume_constants.SLICES: (
                    [
                        {
                            volume_constants.INDEX: 3,
                            volume_constants.FIGURES: [
                                {
                                    volume_constants.ID: 3389398,
                                    volume_constants.OBJECT_ID: 1265,
                                    ApiField.GEOMETRY_TYPE: Rectangle.geometry_name(),
                                    ApiField.GEOMETRY: GEOMETRY,
                                }
                            ],
                        }
                    ]
                    if name == "axial"
                    else []
                ),
            }
            for name, normal in (
                ("sagittal", {"x": 1, "y": 0, "z": 0}),
                ("coronal", {"x": 0, "y": 1, "z": 0}),
                ("axial", {"x": 0, "y": 0, "z": 1}),
            )
        ],
        volume_constants.VOLUME_META: {"dimensionsIJK": {"x": 64, "y": 64, "z": 32}},
        volume_constants.TAGS: [],
    }

    key_id_map = KeyIdMap()
    parsed = VolumeAnnotation.from_json(annotation, META, key_id_map)
    stored = parsed.to_json(key_id_map)

    figure = stored[volume_constants.PLANES][2][volume_constants.SLICES][0][
        volume_constants.FIGURES
    ][0]
    assert figure[volume_constants.ID] == 3389398
    assert figure[volume_constants.OBJECT_ID] == 1265
    # The key stays: it is what binds a figure to its object when a version is restored,
    # and restore parses without a key map to resolve ids through.
    assert volume_constants.KEY in figure
    assert volume_constants.OBJECT_KEY in figure


def test_a_volume_snapshot_says_when_its_figures_cannot_be_matched(tmp_path):
    """Written before the fix above: v2.1.0 in the header, and not one figure id in the
    records. The flag has to follow the records, or a comparison pairs on keys that differ
    by construction and calls the result a diff."""
    keyless = {
        volume_constants.TAGS: [],
        volume_constants.OBJECTS: [
            {volume_constants.KEY: "obj-1", LabelJsonFields.OBJ_CLASS_NAME: "car"}
        ],
        volume_constants.PLANES: [
            {
                volume_constants.NAME: "axial",
                volume_constants.SLICES: [
                    {
                        volume_constants.INDEX: 3,
                        volume_constants.FIGURES: [
                            {
                                volume_constants.KEY: "fig-1",
                                volume_constants.OBJECT_KEY: "obj-1",
                                ApiField.GEOMETRY_TYPE: Rectangle.geometry_name(),
                                ApiField.GEOMETRY: GEOMETRY,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    with VersionSnapshot.open_archive(_volume_path(tmp_path, "keyless.bin", keyless)) as snapshot:
        assert snapshot.schema_version == "v2.1.0"
        assert snapshot.figure_ids_are_server_ids is False

    with_ids = json.loads(json.dumps(keyless))
    figure = with_ids[volume_constants.PLANES][0][volume_constants.SLICES][0][
        volume_constants.FIGURES
    ][0]
    figure[volume_constants.ID] = 5001
    with VersionSnapshot.open_archive(_volume_path(tmp_path, "with-ids.bin", with_ids)) as snapshot:
        assert snapshot.figure_ids_are_server_ids is True


def test_a_volume_snapshot_is_columnar_without_serving_arrow(volume_snapshot):
    """Two different questions. The payload streams section by section rather than being
    loaded whole - columnar by that measure - and still holds one JSON record per volume,
    so there is no column to project."""
    assert volume_snapshot.is_columnar is True
    assert volume_snapshot.serves_arrow is False
    with pytest.raises(NotImplementedError):
        _collect(volume_snapshot.iter_items_arrow())


def test_reading_one_volume_does_not_parse_the_others(volume_snapshot, monkeypatch):
    """The cost of this format is parsing the records, so a filter that runs after the
    parse is no filter at all. Measured on a 3009-volume project: one changed item, and
    five of the diff's six seconds went on reading the other 3008."""
    from supervisely.project.versioning import image_snapshot_io

    parsed = []
    original = image_snapshot_io.loads_or

    def counting(raw, default):
        parsed.append(raw)
        return original(raw, default)

    monkeypatch.setattr(image_snapshot_io, "loads_or", counting)

    _collect(volume_snapshot.iter_tags(item_ids={999}))
    assert parsed == []

    _collect(volume_snapshot.iter_tags(item_ids={20}))
    assert parsed


def test_volume_objects_are_flattened_out_of_the_annotation(volume_snapshot):
    """Same interface, different storage: this format keeps whole records, so the objects
    are read out of the annotation document rather than off a table."""
    objects = _collect(volume_snapshot.iter_objects())

    assert len(objects) == 1
    assert objects[0][SnapshotColumn.OBJECT_ID] == "obj-1"
    assert objects[0][SnapshotColumn.CLASS_NAME] == "car"
    assert objects[0][SnapshotColumn.ITEM_ID] == 20


def _video_writer_api(video_count=2, objects=2, frames=3):
    """A stub API just rich enough to drive VideoProject.build_snapshot."""
    from supervisely.api.dataset_api import DatasetInfo
    from supervisely.api.module_api import ApiField
    from supervisely.api.project_api import ProjectInfo
    from supervisely.api.video.video_api import VideoInfo

    def make(cls, **v):
        return cls(**{f: v.get(f) for f in cls._fields})

    counter = [0]

    def next_id():
        counter[0] += 1
        return counter[0]

    def ann_json(name, video_id):
        """The shape the server sends: objects carry ids, figures point at their parent
        by that id, and nothing carries a uuid key."""
        object_ids = [next_id() for _ in range(objects)]
        return {
            "size": {"height": 100, "width": 200},
            video_constants.FRAMES_COUNT: frames,
            LabelJsonFields.TAGS: [],
            video_constants.OBJECTS: [
                {
                    video_constants.ID: oid,
                    LabelJsonFields.OBJ_CLASS_NAME: "car",
                    LabelJsonFields.TAGS: [],
                }
                for oid in object_ids
            ],
            video_constants.FRAMES: [
                {
                    video_constants.INDEX: i,
                    video_constants.FIGURES: [
                        {
                            video_constants.ID: next_id(),
                            video_constants.OBJECT_ID: oid,
                            ApiField.GEOMETRY_TYPE: Rectangle.geometry_name(),
                            ApiField.GEOMETRY: GEOMETRY,
                        }
                        for oid in object_ids
                    ],
                }
                for i in range(frames)
            ],
            ApiField.VIDEO_NAME: name,
            video_constants.VIDEO_ID: video_id,
        }

    dataset = make(DatasetInfo, id=1, name="clips", parent_id=None)

    class _Annotation:
        def download_bulk(self, ds_id, ids):
            return [ann_json(f"clip{i}.mp4", i) for i in ids]

    class _Video:
        annotation = _Annotation()

        def get_list(self, ds_id):
            return [
                make(VideoInfo, id=i, name=f"clip{i}.mp4", hash=f"h{i}", dataset_id=ds_id,
                     frames_count=frames, frame_width=200, frame_height=100)
                for i in range(video_count)
            ]

        def get_list_generator(self, ds_id, batch_size=None):
            videos = self.get_list(ds_id)
            step = batch_size or len(videos) or 1
            for start in range(0, len(videos), step):
                yield videos[start : start + step]

    class _Project:
        def get_info_by_id(self, pid):
            return make(ProjectInfo, id=pid, name="clips", type="videos")

        def get_meta(self, pid, with_settings=False):
            return META.to_json()

    class _Dataset:
        def get_list(self, pid, recursive=False, include_custom_data=False):
            return [dataset]

        def tree(self, pid):
            yield [], dataset

    class _Api:
        project = _Project()
        dataset = _Dataset()
        video = _Video()

    return _Api(), video_count, video_count * objects, video_count * objects * frames


def test_video_writer_streams_into_the_shared_container(tmp_path):
    """The writer used to hold every row of the project before writing one."""
    from supervisely.project.versioning.container import is_tar_container, is_zstd_container
    from supervisely.project.video_project import VideoProject

    api, videos, objects, figures = _video_writer_api()
    blob = VideoProject.build_snapshot(api, project_id=1, batch_size=1, log_progress=False)
    data = blob.getvalue()

    # A plain tar of already-compressed Parquet, not a second pass over it.
    assert is_tar_container(data)
    assert not is_zstd_container(data)

    path = os.path.join(str(tmp_path), "version.bin")
    with open(path, "wb") as f:
        f.write(data)

    with VersionSnapshot.open_archive(path) as snapshot:
        assert snapshot.project_type == "videos"
        items = _collect(snapshot.iter_items())
        assert len(items) == videos
        assert {i[SnapshotColumn.WIDTH] for i in items} == {200}
        figure_rows = _collect(snapshot.iter_figures())
        assert len(figure_rows) == figures
        assert {f[SnapshotColumn.CLASS_NAME] for f in figure_rows} == {"car"}
        assert {f[SnapshotColumn.FRAME_INDEX] for f in figure_rows} == {0, 1, 2}


def test_video_snapshots_carry_server_ids_and_no_key_id_map(tmp_path):
    """v2.1.0: the ids in the tables are the server's own, and the file nobody read is gone."""
    import tarfile

    from supervisely.project.versioning.common import VIDEO_SCHEMA_VERSION_V2_1
    from supervisely.project.video_project import VideoProject

    api, videos, objects, figures = _video_writer_api(video_count=2)
    data = VideoProject.build_snapshot(api, project_id=1, log_progress=False).getvalue()

    with tarfile.open(fileobj=__import__("io").BytesIO(data), mode="r") as tar:
        names = {m.name.lstrip("./") for m in tar.getmembers() if m.isfile()}
        manifest = json.loads(tar.extractfile("./manifest.json").read())

    assert "key_id_map.json" not in names
    assert manifest[VersionSchemaField.SCHEMA_VERSION] == VIDEO_SCHEMA_VERSION_V2_1

    path = os.path.join(str(tmp_path), "version.bin")
    with open(path, "wb") as f:
        f.write(data)

    with VersionSnapshot.open_archive(path) as snapshot:
        rows = _collect(snapshot.iter_figures())
        ids = [r[SnapshotColumn.FIGURE_ID] for r in rows]

    # Server ids, not 1..N row counters - that is the whole point of the bump.
    assert len(ids) == figures
    assert len(set(ids)) == figures
    # The stub hands out one id sequence across objects and figures, so a row counter
    # would have produced exactly 1..N. Anything else means the server ids came through.
    assert sorted(ids) != list(range(1, figures + 1))


def _legacy_video_snapshot(tmp_path, api, video_count=2):
    """A genuine v2.0.0 snapshot: ann_json on the video row, tables numbered by position."""
    from supervisely.project.versioning.common import (
        VIDEO_SCHEMA_VERSION_V2,
        get_video_snapshot_schema,
    )
    from supervisely.project.versioning.container import (
        ParquetTableWriter,
        pack_payload_dir,
        table_meta,
    )

    schema = get_video_snapshot_schema(VIDEO_SCHEMA_VERSION_V2)
    payload = os.path.join(str(tmp_path), "legacy_payload")
    os.makedirs(payload, exist_ok=True)
    dump_json_file(
        {"id": 1, "name": "clips", "type": "videos"}, os.path.join(payload, "project_info.json")
    )
    dump_json_file(META.to_json(), os.path.join(payload, "project_meta.json"))

    writers = {
        name: ParquetTableWriter(
            pyarrow, parquet, os.path.join(payload, f"{name}.parquet"), table_schema
        )
        for name, table_schema in (
            ("videos", schema.videos_schema(pyarrow)),
            ("objects", schema.objects_schema(pyarrow)),
            ("figures", schema.figures_schema(pyarrow)),
        )
    }

    videos = api.video.get_list(1)[:video_count]
    anns = api.video.annotation.download_bulk(1, [v.id for v in videos])
    obj_n = fig_n = 0
    for video_info, ann in zip(videos, anns):
        writers["videos"].add(
            schema.video_row_from_video_info(video_info, src_dataset_id=1, ann_json=ann)
        )
        positions = {}
        for obj in ann[video_constants.OBJECTS]:
            obj_n += 1
            positions[obj[video_constants.ID]] = obj_n
            writers["objects"].add(
                {
                    VersionSchemaField.SRC_OBJECT_ID: obj_n,
                    VersionSchemaField.SRC_VIDEO_ID: video_info.id,
                    VersionSchemaField.CLASS_NAME: obj[LabelJsonFields.OBJ_CLASS_NAME],
                    VersionSchemaField.KEY: "deadbeef",
                    VersionSchemaField.TAGS_JSON: None,
                }
            )
        for frame in ann[video_constants.FRAMES]:
            for fig in frame[video_constants.FIGURES]:
                fig_n += 1
                writers["figures"].add(
                    {
                        VersionSchemaField.SRC_FIGURE_ID: fig_n,
                        VersionSchemaField.SRC_OBJECT_ID: positions[
                            fig[video_constants.OBJECT_ID]
                        ],
                        VersionSchemaField.SRC_VIDEO_ID: video_info.id,
                        VersionSchemaField.FRAME_INDEX: frame[video_constants.INDEX],
                        VersionSchemaField.GEOMETRY_TYPE: fig[ApiField.GEOMETRY_TYPE],
                        VersionSchemaField.GEOMETRY_JSON: json.dumps(fig[ApiField.GEOMETRY]),
                    }
                )

    tables_meta = []
    for name, writer in writers.items():
        count = writer.close()
        if count:
            tables_meta.append(table_meta(name, f"{name}.parquet", count))
    dump_json_file(
        {
            VersionSchemaField.SCHEMA_VERSION: VIDEO_SCHEMA_VERSION_V2,
            VersionSchemaField.TABLES: tables_meta,
        },
        os.path.join(payload, "manifest.json"),
    )
    path = os.path.join(str(tmp_path), "legacy.bin")
    with open(path, "wb") as f:
        f.write(pack_payload_dir(payload, str(tmp_path)).getvalue())
    return path


def test_legacy_video_snapshots_read_but_say_their_ids_cannot_be_matched(tmp_path):
    api, *_ = _video_writer_api(video_count=2)
    path = _legacy_video_snapshot(tmp_path, api)

    with VersionSnapshot.open_archive(path) as snapshot:
        assert snapshot.schema_version == "v2.0.0"
        rows = _collect(snapshot.iter_figures())
        # Readable, and the class still resolves through the objects table...
        assert {r[SnapshotColumn.CLASS_NAME] for r in rows} == {"car"}
        # ...but the ids are positions, and the reader says so rather than letting a
        # comparison pair unrelated figures.
        assert sorted(r[SnapshotColumn.FIGURE_ID] for r in rows) == list(range(1, len(rows) + 1))
        assert snapshot.figure_ids_are_server_ids is False


def test_repacking_a_legacy_video_snapshot_recovers_the_server_ids(tmp_path):
    """The ids were never lost - ann_json had them all along."""
    from supervisely.project.video_project import VideoProject

    api, *_ = _video_writer_api(video_count=2)
    legacy = _legacy_video_snapshot(tmp_path, api)

    with VersionSnapshot.open_archive(legacy) as snapshot:
        legacy_rows = _collect(snapshot.iter_figures())

    repacked_path = os.path.join(str(tmp_path), "repacked.bin")
    with open(repacked_path, "wb") as f:
        f.write(VideoProject.repack_snapshot(legacy).getvalue())

    with VersionSnapshot.open_archive(repacked_path) as snapshot:
        assert snapshot.schema_version == "v2.1.0"
        assert snapshot.figure_ids_are_server_ids is True
        rows = _collect(snapshot.iter_figures())

    assert len(rows) == len(legacy_rows)
    ids = sorted(r[SnapshotColumn.FIGURE_ID] for r in rows)
    assert ids != list(range(1, len(rows) + 1))
    # Same annotations, same classes and frames - only the identity got better.
    assert {r[SnapshotColumn.CLASS_NAME] for r in rows} == {"car"}
    assert sorted(r[SnapshotColumn.FRAME_INDEX] for r in rows) == sorted(
        r[SnapshotColumn.FRAME_INDEX] for r in legacy_rows
    )


def test_volume_snapshots_report_whether_their_ids_are_the_servers(tmp_path):
    """Header version 1 kept only uuid keys, invented per parse; version 2 keeps ids."""
    from supervisely.project.volume_project import VolumeProject

    def blob_with(version, annotation):
        payload = VolumeProject._assemble_sections(
            [
                (
                    VolumeProject._SECTION_PROJECT_INFO,
                    json.dumps({"id": 1, "name": "p", "type": "volumes"}).encode(),
                ),
                (VolumeProject._SECTION_PROJECT_META, json.dumps(META.to_json()).encode()),
                (
                    VolumeProject._SECTION_ANNOTATIONS,
                    _parquet_bytes(
                        pyarrow.Table.from_pylist(
                            [
                                get_volume_snapshot_schema("v2.0.0").annotation_row_from_dict(
                                    src_volume_id=20, annotation=annotation
                                )
                            ],
                            schema=get_volume_snapshot_schema("v2.0.0").annotations_table_schema(
                                pyarrow
                            ),
                        )
                    ),
                ),
            ]
        )
        # _assemble_sections stamps the current header version; rewrite it to test the old one.
        magic = VolumeProject._SERIALIZATION_MAGIC
        return payload[: len(magic)] + bytes([version]) + payload[len(magic) + 1 :]

    def annotation(with_ids):
        figure = {
            volume_constants.KEY: "fig-1",
            volume_constants.OBJECT_KEY: "obj-1",
            ApiField.GEOMETRY_TYPE: Rectangle.geometry_name(),
            ApiField.GEOMETRY: GEOMETRY,
        }
        if with_ids:
            figure[volume_constants.ID] = 5001
        return {
            volume_constants.OBJECTS: [
                {volume_constants.KEY: "obj-1", LabelJsonFields.OBJ_CLASS_NAME: "car"}
            ],
            volume_constants.PLANES: [
                {
                    volume_constants.NAME: "axial",
                    volume_constants.SLICES: [
                        {volume_constants.INDEX: 0, volume_constants.FIGURES: [figure]}
                    ],
                }
            ],
        }

    legacy = os.path.join(str(tmp_path), "v1.bin")
    with open(legacy, "wb") as f:
        f.write(blob_with(1, annotation(with_ids=False)))
    with VersionSnapshot.open_archive(legacy) as snapshot:
        assert snapshot.figure_ids_are_server_ids is False
        assert _collect(snapshot.iter_figures())[0][SnapshotColumn.FIGURE_ID] == "fig-1"

    current = os.path.join(str(tmp_path), "v2.bin")
    with open(current, "wb") as f:
        f.write(blob_with(2, annotation(with_ids=True)))
    with VersionSnapshot.open_archive(current) as snapshot:
        assert snapshot.figure_ids_are_server_ids is True
        assert _collect(snapshot.iter_figures())[0][SnapshotColumn.FIGURE_ID] == 5001


class _RestoreProjectApi:
    def __init__(self):
        self.created = None

    def exists(self, workspace_id, name):
        return False

    def create(self, workspace_id, name, *args, **kwargs):
        self.created = SimpleNamespace(
            id=900, name=name, custom_data={}, description=kwargs.get("description")
        )
        return self.created

    def update_meta(self, project_id, meta):
        return meta if isinstance(meta, ProjectMeta) else ProjectMeta.from_json(meta)

    def update_custom_data(self, *args, **kwargs):
        return None

    def get_info_by_id(self, project_id):
        return self.created


class _RestoreDatasetApi:
    def __init__(self):
        self.created = []

    def create(self, project_id, name, **kwargs):
        info = SimpleNamespace(id=1000 + len(self.created), name=name)
        self.created.append(info)
        return info


def _restore_api():
    return SimpleNamespace(
        project=_RestoreProjectApi(), dataset=_RestoreDatasetApi(), optimization_context={}
    )


def test_old_video_v2_snapshot_is_still_restorable(tmp_path):
    """The old ann_json schema remains a restore input after v2.1 becomes the writer."""
    from supervisely.project.video_project import VideoProject

    schema = get_video_snapshot_schema("v2.0.0")
    path = _write_payload(
        tmp_path,
        "videos",
        {
            "datasets": (
                schema.datasets_schema(pyarrow),
                [
                    schema.dataset_row(
                        src_dataset_id=1,
                        parent_src_dataset_id=None,
                        name="legacy",
                        full_path="legacy",
                        description=None,
                        custom_data=None,
                    )
                ],
            )
        },
    )
    api = _restore_api()

    restored = VideoProject.upload_bin(
        api, path, workspace_id=1, log_progress=False, restore_workers=1
    )

    assert restored.id == 900
    assert [dataset.name for dataset in api.dataset.created] == ["legacy"]


def test_old_volume_v2_snapshot_is_still_restorable(tmp_path):
    """Header version 1 is the deployed volume v2 format and stays accepted by upload_bin."""
    from supervisely.project.volume_project import VolumeProject

    schema = get_volume_snapshot_schema("v2.0.0")
    dataset_table = pyarrow.Table.from_pylist(
        [schema.dataset_row_from_record({"id": 1, "name": "legacy"})],
        schema=schema.datasets_table_schema(pyarrow),
    )
    payload = VolumeProject._assemble_sections(
        [
            (
                VolumeProject._SECTION_PROJECT_INFO,
                json.dumps({"id": 1, "name": "p", "type": "volumes"}).encode(),
            ),
            (VolumeProject._SECTION_PROJECT_META, json.dumps(META.to_json()).encode()),
            (VolumeProject._SECTION_DATASETS, _parquet_bytes(dataset_table)),
        ]
    )
    magic = VolumeProject._SERIALIZATION_MAGIC
    payload = payload[: len(magic)] + b"\x01" + payload[len(magic) + 1 :]
    path = os.path.join(str(tmp_path), "legacy-volume.bin")
    with open(path, "wb") as stream:
        stream.write(payload)
    api = _restore_api()

    restored = VolumeProject.upload_bin(
        api,
        path,
        workspace_id=1,
        log_progress=False,
        restore_workers=1,
        with_custom_data=False,
    )

    assert restored.id == 900
    assert [dataset.name for dataset in api.dataset.created] == ["legacy"]


def _rebuild_annotations(archive_path):
    """What restore_snapshot now does: annotation per video, out of the tables."""
    import tempfile

    from supervisely.project.versioning.container import unpack_snapshot
    from supervisely.project.versioning.video_schema import annotation_json_from_rows

    out = {}
    with tempfile.TemporaryDirectory() as payload:
        unpack_snapshot(open(archive_path, "rb").read(), payload)
        from supervisely.project.versioning.tag_schema import OWNER_ITEM, tag_json_from_row

        objects, figures = {}, {}
        for table, sink in (("objects.parquet", objects), ("figures.parquet", figures)):
            path = os.path.join(payload, table)
            if os.path.exists(path):
                for row in parquet.read_table(path).to_pylist():
                    sink.setdefault(row[VersionSchemaField.SRC_VIDEO_ID], []).append(row)

        item_tags, object_tags = {}, {}
        tags_path = os.path.join(payload, "tags.parquet")
        if os.path.exists(tags_path):
            for row in parquet.read_table(tags_path).to_pylist():
                sink = (
                    item_tags if row[VersionSchemaField.OWNER_TYPE] == OWNER_ITEM else object_tags
                )
                sink.setdefault(row[VersionSchemaField.OWNER_ID], []).append(
                    tag_json_from_row(row)
                )

        for row in parquet.read_table(os.path.join(payload, "videos.parquet")).to_pylist():
            vid = row[VersionSchemaField.SRC_VIDEO_ID]
            out[vid] = annotation_json_from_rows(
                row,
                objects.get(vid, []),
                figures.get(vid, []),
                item_tags=item_tags.get(vid, []),
                object_tags=object_tags,
            )
    return out


def test_the_annotation_rebuilt_from_tables_matches_what_the_server_sent(tmp_path):
    """v2.1.0 drops ann_json, so the tables have to be able to reproduce it. Everything
    except the identity fields must come back unchanged - ids belong to the source
    project and keys are minted fresh, because a restore uploads into a new project."""
    from supervisely.project.video_project import VideoProject

    api, *_ = _video_writer_api(video_count=2, objects=2, frames=3)
    videos = api.video.get_list(1)[:2]
    source = dict(zip([v.id for v in videos],
                      api.video.annotation.download_bulk(1, [v.id for v in videos])))

    path = os.path.join(str(tmp_path), "version.bin")
    with open(path, "wb") as f:
        f.write(VideoProject.build_snapshot(api, project_id=1, log_progress=False).getvalue())

    # The stub hands out a fresh id sequence per call, so compare against a snapshot of
    # the same shape rather than the same ids.
    rebuilt = _rebuild_annotations(path)
    assert set(rebuilt) == set(source)

    for video_id, original in source.items():
        got = rebuilt[video_id]
        assert got[volume_constants.IMG_SIZE] == original["size"]
        assert got[video_constants.FRAMES_COUNT] == original[video_constants.FRAMES_COUNT]
        assert got[video_constants.TAGS] == original[LabelJsonFields.TAGS]

        assert [o[LabelJsonFields.OBJ_CLASS_NAME] for o in got[video_constants.OBJECTS]] == [
            o[LabelJsonFields.OBJ_CLASS_NAME] for o in original[video_constants.OBJECTS]
        ]

        assert [f[video_constants.INDEX] for f in got[video_constants.FRAMES]] == [
            f[video_constants.INDEX] for f in original[video_constants.FRAMES]
        ]
        for got_frame, original_frame in zip(
            got[video_constants.FRAMES], original[video_constants.FRAMES]
        ):
            got_figures = got_frame[video_constants.FIGURES]
            original_figures = original_frame[video_constants.FIGURES]
            assert len(got_figures) == len(original_figures)
            for gf, of in zip(got_figures, original_figures):
                assert gf[ApiField.GEOMETRY_TYPE] == of[ApiField.GEOMETRY_TYPE]
                assert gf[ApiField.GEOMETRY] == of[ApiField.GEOMETRY]
                # Linked by key, and the key resolves to the right object.
                keys = {o[video_constants.KEY] for o in got[video_constants.OBJECTS]}
                assert gf[video_constants.OBJECT_KEY] in keys


def test_figure_fields_that_only_ann_json_used_to_carry_survive(tmp_path):
    """track id, priority, smart-tool input and the nn flags: without columns for them a
    restore built from the tables would quietly drop them."""
    from supervisely.project.versioning.common import get_video_snapshot_schema
    from supervisely.project.versioning.video_schema import annotation_json_from_rows

    schema = get_video_snapshot_schema("v2.1.0")
    figure_json = {
        video_constants.ID: 4021,
        video_constants.OBJECT_ID: 88,
        ApiField.GEOMETRY_TYPE: Rectangle.geometry_name(),
        ApiField.GEOMETRY: GEOMETRY,
        ApiField.META: {"frame": 7},
        # A real track id is a uuid string, not a number.
        ApiField.TRACK_ID: "991ad1d4-ced5-434c-9955-a55b352e74ee",
        ApiField.PRIORITY: 3,
        ApiField.SMART_TOOL_INPUT: {"crop": [[1, 2], [3, 4]]},
        ApiField.NN_CREATED: True,
    }
    figure_row = schema.figure_row_from_json(figure_json, src_video_id=10, frame_index=7)
    object_row = schema.object_row_from_json(
        {video_constants.ID: 88, LabelJsonFields.OBJ_CLASS_NAME: "car"}, src_video_id=10
    )
    video_row = {
        VersionSchemaField.FRAME_HEIGHT: 100,
        VersionSchemaField.FRAME_WIDTH: 200,
        VersionSchemaField.FRAMES_COUNT: 8,
    }

    rebuilt = annotation_json_from_rows(video_row, [object_row], [figure_row])
    figure = rebuilt[video_constants.FRAMES][0][video_constants.FIGURES][0]

    assert figure[ApiField.META] == {"frame": 7}
    assert figure[ApiField.TRACK_ID] == "991ad1d4-ced5-434c-9955-a55b352e74ee"
    assert figure[ApiField.PRIORITY] == 3
    assert figure[ApiField.SMART_TOOL_INPUT] == {"crop": [[1, 2], [3, 4]]}
    assert figure[ApiField.NN_CREATED] is True
    # Not set rather than set to false, matching what the server omits.
    assert ApiField.NN_UPDATED not in figure


def test_repack_carries_over_the_video_tags_and_description(tmp_path):
    """The target schema replaced ann_json with two columns of its own; a verbatim row
    copy would have silently dropped both."""
    from supervisely.project.video_project import VideoProject

    api, *_ = _video_writer_api(video_count=1)

    # Give the stub's annotation a video-level tag and a description, the two things that
    # lived only inside ann_json.
    original_download = api.video.annotation.download_bulk

    def with_extras(ds_id, ids):
        anns = original_download(ds_id, ids)
        for ann in anns:
            ann[LabelJsonFields.TAGS] = [{"name": "reviewed", "frameRange": [0, 2]}]
            ann[video_constants.DESCRIPTION] = "checked by hand"
        return anns

    api.video.annotation.download_bulk = with_extras

    legacy = _legacy_video_snapshot(tmp_path, api, video_count=1)
    repacked = os.path.join(str(tmp_path), "repacked.bin")
    with open(repacked, "wb") as f:
        f.write(VideoProject.repack_snapshot(legacy).getvalue())

    rebuilt = _rebuild_annotations(repacked)
    ann = next(iter(rebuilt.values()))
    tag = ann[video_constants.TAGS][0]
    assert tag["name"] == "reviewed"
    assert tag["frameRange"] == [0, 2]
    assert ann[video_constants.DESCRIPTION] == "checked by hand"


def test_tags_can_be_read_as_columns(tmp_path):
    """The tags table is a table like any other, and a comparison that counts tag
    assignments over a whole project has no business building a dict per row.

    `value` is the exception: it is decoded when a row is built, so the columnar path
    serves `value_json` - the column as stored - and says so rather than inventing one.
    """
    from supervisely.project.versioning.tag_schema import OWNER_ITEM, get_tag_schema

    schema = get_tag_schema()
    rows = [
        schema.tag_row(
            {"id": 900 + i, "tagId": 7, "name": "reviewed", "value": i},
            owner_type=OWNER_ITEM,
            owner_id=10 + i,
            src_item_id=10 + i,
        )
        for i in range(3)
    ]
    tables = {"tags": (schema.tags_schema(pyarrow), rows)}
    path = _write_payload(tmp_path, "images", tables)

    with VersionSnapshot.open_archive(path) as snapshot:
        batches = list(
            snapshot.iter_tags_arrow(
                columns=[SnapshotColumn.ITEM_ID, SnapshotColumn.VALUE_JSON]
            )
        )
        table = pyarrow.Table.from_batches(batches)
        assert table.column_names == [SnapshotColumn.ITEM_ID, SnapshotColumn.VALUE_JSON]
        assert table.column(SnapshotColumn.ITEM_ID).to_pylist() == [10, 11, 12]
        assert table.column(SnapshotColumn.VALUE_JSON).to_pylist() == ["0", "1", "2"]

        with pytest.raises(ValueError, match="derived"):
            list(snapshot.iter_tags_arrow(columns=[SnapshotColumn.VALUE]))

        # The dict path still hands back the decoded value.
        assert _collect(snapshot.iter_tags())[0][SnapshotColumn.VALUE] == 0


@pytest.mark.parametrize("chunk", [200, 1])
def test_a_v2_1_video_restore_rebuilds_each_dataset_and_files_it_there(
    tmp_path, monkeypatch, chunk
):
    """Restore reads the tables a chunk of videos at a time and appends with the ids it knows.

    Two datasets, so that a rebuild keyed on the wrong dataset's rows, or an append filed
    under the wrong dataset, shows up as a miscount rather than passing by accident; and a
    chunk of one, so that a dataset split across chunks still restores every video.
    """
    from supervisely.api.dataset_api import DatasetInfo
    from supervisely.api.video.video_api import VideoInfo
    from supervisely.project import video_project
    from supervisely.project.video_project import VideoProject

    monkeypatch.setattr(video_project, "RESTORE_READ_CHUNK", chunk)

    api, _, _, _ = _video_writer_api(video_count=2, objects=2, frames=3)
    datasets = [
        DatasetInfo(**{f: None for f in DatasetInfo._fields})._replace(id=1, name="a"),
        DatasetInfo(**{f: None for f in DatasetInfo._fields})._replace(id=2, name="b"),
    ]
    original_list = api.video.get_list

    def get_list(ds_id):
        # Distinct video ids per dataset, and a different number of videos in each.
        videos = original_list(ds_id)[: ds_id]
        return [v._replace(id=ds_id * 100 + v.id, name=f"{ds_id}-{v.name}") for v in videos]

    api.video.get_list = get_list
    download = api.video.annotation.download_bulk
    api.video.annotation.download_bulk = lambda ds_id, ids: [
        dict(ann, videoName=f"{ds_id}-clip{video_id - ds_id * 100}.mp4")
        for video_id, ann in zip(ids, download(ds_id, ids))
    ]
    api.dataset.get_list = lambda *a, **k: datasets
    api.dataset.tree = lambda pid: iter([([], datasets[0]), ([], datasets[1])])

    path = os.path.join(str(tmp_path), "version.bin")
    with open(path, "wb") as f:
        f.write(VideoProject.build_snapshot(api, project_id=1, log_progress=False).getvalue())

    restore_api = _restore_api()
    appended = []
    info_reads = []

    class _RestoreAnnotation:
        def append(self, video_id, ann, key_id_map=None, progress_cb=None, video_info=None):
            appended.append(
                (video_id, len(ann.figures), video_info.project_id, video_info.dataset_id)
            )

    class _RestoreVideo:
        annotation = _RestoreAnnotation()

        def upload_hashes(self, dataset_id, names, hashes, metas=None, progress_cb=None):
            # videos.bulk.add does not say which project a video landed in.
            return [
                VideoInfo(**{f: None for f in VideoInfo._fields})._replace(
                    id=dataset_id * 10 + i, name=name
                )
                for i, name in enumerate(names)
            ]

        def get_info_by_id(self, video_id):
            info_reads.append(video_id)
            raise AssertionError("restore already knows both ids")

    restore_api.video = _RestoreVideo()
    VideoProject.upload_bin(
        restore_api, path, workspace_id=1, log_progress=False, restore_workers=1
    )

    new_dataset_ids = [d.id for d in restore_api.dataset.created]
    # Dataset "a" has one video, "b" two; each video carries 2 objects x 3 frames.
    assert sorted(appended) == sorted(
        [(new_dataset_ids[0] * 10, 6, 900, new_dataset_ids[0])]
        + [(new_dataset_ids[1] * 10 + i, 6, 900, new_dataset_ids[1]) for i in range(2)]
    )
    assert info_reads == []


def test_volume_datasets_carry_their_full_path(tmp_path):
    """A volume snapshot stores no dataset path; without one two nested datasets that share
    a leaf name are the same dataset to anything matching on paths."""
    from supervisely.project.volume_project import VolumeProject

    schema = get_volume_snapshot_schema("v2.1.0")
    datasets = pyarrow.Table.from_pylist(
        [
            # What the writer stores: DatasetInfo._asdict(), snake_case keys.
            schema.dataset_row_from_record(_dataset_record(1, "a")),
            schema.dataset_row_from_record(_dataset_record(2, "b")),
            schema.dataset_row_from_record(_dataset_record(3, "train", parent_id=1)),
            schema.dataset_row_from_record(
                _dataset_record(4, "train", parent_id=2, custom_data={"k": 1})
            ),
        ],
        schema=schema.datasets_table_schema(pyarrow),
    )
    blob = VolumeProject._assemble_sections(
        [
            (
                VolumeProject._SECTION_PROJECT_INFO,
                json.dumps({"id": 1, "name": "p", "type": "volumes"}).encode(),
            ),
            (VolumeProject._SECTION_PROJECT_META, json.dumps(META.to_json()).encode()),
            (VolumeProject._SECTION_DATASETS, _parquet_bytes(datasets)),
        ]
    )
    path = os.path.join(str(tmp_path), "volume.bin")
    with open(path, "wb") as f:
        f.write(blob)

    with VersionSnapshot.open_archive(path) as snapshot:
        paths = {ds.id: ds.full_path for ds in snapshot.datasets()}

    assert paths[3] != paths[4]
    assert paths[1] == "a" and paths[2] == "b"
    assert paths[3].endswith("train") and paths[3].startswith("a")
    with VersionSnapshot.open_archive(path) as snapshot:
        by_id = {ds.id: ds for ds in snapshot.datasets()}
    assert by_id[4].parent_id == 2 and by_id[4].custom_data == {"k": 1}


def _dataset_record(dataset_id, name, parent_id=None, custom_data=None):
    from supervisely.api.dataset_api import DatasetInfo

    info = DatasetInfo(**{f: None for f in DatasetInfo._fields})._replace(
        id=dataset_id, name=name, parent_id=parent_id, custom_data=custom_data
    )
    return info._asdict()
