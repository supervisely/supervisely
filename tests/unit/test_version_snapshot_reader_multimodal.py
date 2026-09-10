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
                }
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


# ------------------------------------------------------- the video writer itself


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
