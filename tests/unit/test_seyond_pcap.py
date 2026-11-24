"""Seyond lidar pcap captures: the frame index, the import config, the converters' detection.

Every capture here is synthetic: Falcon packets built byte by byte the way the labeling tool's own
tests build them (one block per packet), wrapped in the link layers tcpdump writes.
"""

import json
import os
import struct
import types

import numpy as np
import pytest

from supervisely.convert.pointcloud.pointcloud_converter import PointcloudConverter
from supervisely.convert.pointcloud.seyond_pcap.seyond_pcap_converter import (
    SeyondPcapConverter,
)
from supervisely.convert.pointcloud.seyond_pcap.seyond_pcap_helper import (
    SensorsConfig,
    parse_capture_name,
    sensor_key,
    to_extrinsic_matrix,
)
from supervisely.convert.pointcloud_episodes.seyond_pcap.seyond_pcap_converter import (
    SeyondPcapEpisodesConverter,
)
from supervisely.convert.converter import ImportManager
from supervisely.pointcloud.pointcloud import is_valid_ext
from supervisely.pointcloud.seyond_pcap import (
    SeyondPcapError,
    index_pcap,
    to_iso_timestamp,
)
from supervisely.project.pointcloud_project import _download_pointcloud_file
from supervisely.project.project_type import ProjectType

ROLL_MINUS_90 = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0]


def seyond_packet(frame_idx, item_type=1, version=2, lidar_type=3, size=None):
    header_size = 54 if version == 1 else 70
    packet = bytearray(header_size)
    packet[0:2] = struct.pack("<H", 0x176A)
    packet[2] = version
    packet[15] = lidar_type
    packet[26:34] = struct.pack("<Q", frame_idx)
    packet[38:42] = struct.pack("<I", item_type | (1 << 8))
    packet += bytes(17 + 16)  # one block: header and 4 channel points
    return bytes(packet if size is None else packet[:size])


def ipv4_udp(payload, fragment=0):
    ip = bytearray(20)
    ip[0] = 0x45
    ip[2:4] = struct.pack(">H", 20 + 8 + len(payload))
    ip[6:8] = struct.pack(">H", fragment)
    ip[9] = 17
    udp = bytearray(8)
    udp[4:6] = struct.pack(">H", 8 + len(payload))
    return bytes(ip + udp) + payload


def ipv6_udp(payload):
    ip = bytearray(40)
    ip[0] = 0x60
    ip[4:6] = struct.pack(">H", 8 + len(payload))
    ip[6] = 17
    udp = bytearray(8)
    udp[4:6] = struct.pack(">H", 8 + len(payload))
    return bytes(ip + udp) + payload


def ethernet(ip_packet, ether_type=0x0800, vlans=()):
    frame = bytearray(12)
    for tpid in vlans:
        frame += struct.pack(">HH", tpid, 7)
    frame += struct.pack(">H", ether_type)
    return bytes(frame) + ip_packet


def capture(path, frames, link_type=1, order="<", nanoseconds=False, start=1_700_000_000.0):
    """Write a classic pcap with one record per frame payload, 10 ms apart."""
    magic = 0xA1B23C4D if nanoseconds else 0xA1B2C3D4
    with open(path, "wb") as f:
        f.write(struct.pack(order + "IHHiIII", magic, 2, 4, 0, 0, 65535, link_type))
        for i, frame in enumerate(frames):
            t = start + i * 0.01
            frac = int(round((t % 1) * (1e9 if nanoseconds else 1e6)))
            f.write(struct.pack(order + "IIII", int(t), frac, len(frame), len(frame)))
            f.write(frame)
    return path


def falcon_capture(path, frame_indices, **kwargs):
    records = [ethernet(ipv4_udp(seyond_packet(i))) for i in frame_indices]
    return capture(path, records, **kwargs)


# ---- the frame index ----


def test_frames_in_the_order_they_start(tmp_path):
    path = falcon_capture(tmp_path / "a.pcap", [8, 7, 8, 9, 9, 9])
    index = index_pcap(path)
    assert [f.lidar_frame_idx for f in index.frames] == [8, 7, 9]
    assert [f.index for f in index.frames] == [0, 1, 2]
    assert [f.packets for f in index.frames] == [2, 1, 3]
    assert index.frames[1].time == pytest.approx(1_700_000_000.01, abs=1e-6)
    assert index.unsupported_reason() is None
    assert index.is_seyond


@pytest.mark.parametrize(
    "make_record, link_type",
    [
        (lambda p: ethernet(ipv4_udp(p)), 1),
        (lambda p: ethernet(ipv4_udp(p), vlans=(0x8100,)), 1),
        (lambda p: ethernet(ipv4_udp(p), vlans=(0x88A8, 0x8100)), 1),
        (lambda p: ethernet(ipv6_udp(p), ether_type=0x86DD), 1),
        (lambda p: ipv4_udp(p), 101),
        (lambda p: ipv6_udp(p), 101),
        (lambda p: bytes(14) + struct.pack(">H", 0x0800) + ipv4_udp(p), 113),
        (lambda p: struct.pack(">H", 0x0800) + bytes(18) + ipv4_udp(p), 276),
    ],
)
def test_link_layers(tmp_path, make_record, link_type):
    records = [make_record(seyond_packet(i)) for i in (1, 1, 2)]
    index = index_pcap(capture(tmp_path / "a.pcap", records, link_type=link_type))
    assert index.frame_count == 2


@pytest.mark.parametrize("order, nanoseconds", [("<", False), (">", False), ("<", True), (">", True)])
def test_byte_order_and_nanoseconds(tmp_path, order, nanoseconds):
    path = falcon_capture(tmp_path / "a.pcap", [1, 2], order=order, nanoseconds=nanoseconds)
    index = index_pcap(path)
    assert index.frame_count == 2
    assert index.frames[1].time == pytest.approx(1_700_000_000.01, abs=1e-6)


def test_protocol_v1_header(tmp_path):
    records = [ethernet(ipv4_udp(seyond_packet(i, version=1))) for i in (5, 6)]
    assert index_pcap(capture(tmp_path / "a.pcap", records)).frame_count == 2


def test_fragmented_datagrams_are_recognised_but_not_decoded(tmp_path):
    # the labeling tool does not reassemble IP fragments, so neither does the index
    records = [
        ethernet(ipv4_udp(seyond_packet(1), fragment=0x2000)),  # first fragment, more follow
        ethernet(bytes([0x45]) + bytes(5) + struct.pack(">H", 0x0010) + bytes([0, 17]) + bytes(10) + b"x" * 40),
        ethernet(ipv4_udp(seyond_packet(2))),
    ]
    index = index_pcap(capture(tmp_path / "a.pcap", records))
    assert [f.lidar_frame_idx for f in index.frames] == [2]
    assert index.fragmented_packets == 1
    assert index.seyond_packets == 2


def test_other_lidar_models_are_named_and_refused(tmp_path):
    records = [ethernet(ipv4_udp(seyond_packet(i, item_type=7, lidar_type=1))) for i in (1, 2)]
    index = index_pcap(capture(tmp_path / "a.pcap", records))
    assert index.frame_count == 0 and index.is_seyond
    assert index.model == "Robin W"
    reason = index.unsupported_reason()
    assert "Robin W" in reason and "item type 7" in reason


def test_packets_too_short_and_other_traffic_are_skipped(tmp_path):
    records = [
        ethernet(ipv4_udp(seyond_packet(1, size=60))),  # v2 header needs 70 bytes
        ethernet(ipv4_udp(b"\x00" * 100)),
        ethernet(b"\x00" * 30, ether_type=0x0806),  # ARP
        ethernet(ipv4_udp(seyond_packet(3))),
    ]
    index = index_pcap(capture(tmp_path / "a.pcap", records))
    assert [f.lidar_frame_idx for f in index.frames] == [3]


def test_a_capture_cut_mid_record_keeps_what_came_before(tmp_path):
    path = falcon_capture(tmp_path / "a.pcap", [1, 2, 3])
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 10)
    index = index_pcap(path)
    assert index.frame_count == 2 and index.truncated


def test_a_capture_cut_mid_record_header_is_truncated(tmp_path):
    path = falcon_capture(tmp_path / "a.pcap", [1, 2])
    with open(path, "ab") as f:
        f.write(bytes(7))
    index = index_pcap(path)
    assert index.frame_count == 2 and index.truncated
    assert not index_pcap(falcon_capture(tmp_path / "b.pcap", [1, 2])).truncated


def test_not_pcap(tmp_path):
    pcapng = tmp_path / "a.pcapng.pcap"
    pcapng.write_bytes(b"\x0a\x0d\x0d\x0a" + bytes(40))
    with pytest.raises(SeyondPcapError, match="pcapng"):
        index_pcap(pcapng)
    other = tmp_path / "b.pcap"
    other.write_bytes(b"not a capture at all")
    with pytest.raises(SeyondPcapError):
        index_pcap(other)


def test_timestamps():
    assert to_iso_timestamp(1787788079.969898) == "2026-08-26T23:47:59.969Z"
    assert to_iso_timestamp(0.0) == "1970-01-01T00:00:00.000Z"


def test_pcap_is_a_point_cloud_extension():
    assert is_valid_ext(".pcap") and is_valid_ext(".PCAP")


# ---- the import config ----


def test_capture_names():
    name = "truck-7_forwardRight_falcon_2026-08-26--16-48-00.pcap"
    assert parse_capture_name(name) == {
        "device": "truck-7",
        "location": "forwardRight",
        "model": "falcon",
        "timestamp": "2026-08-26--16-48-00",
    }
    assert sensor_key(name) == "truck-7_forwardRight"
    assert sensor_key("/data/recording.pcap") == "recording"


def test_matrices():
    assert to_extrinsic_matrix(ROLL_MINUS_90) == ROLL_MINUS_90
    assert to_extrinsic_matrix(ROLL_MINUS_90 + [0, 0, 0, 1]) == ROLL_MINUS_90
    assert to_extrinsic_matrix({"xyz": [0, 0, 0], "rpyDegrees": [-90, 0, 0]}) == pytest.approx(
        ROLL_MINUS_90, abs=1e-9
    )
    m = np.array(to_extrinsic_matrix({"xyz": [1, 2, 3], "rpy": [0.1, 0.2, 0.3]})).reshape(3, 4)
    assert m[:, 3].tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="rotation"):
        to_extrinsic_matrix([2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])  # scale
    with pytest.raises(ValueError, match="rotation"):
        to_extrinsic_matrix([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])  # mirror
    with pytest.raises(ValueError, match="12 or 16"):
        to_extrinsic_matrix([1, 0, 0])
    with pytest.raises(ValueError, match="0 0 0 1"):
        to_extrinsic_matrix(ROLL_MINUS_90 + [0, 0, 1, 1])


def test_sensors_config(tmp_path):
    path = tmp_path / "sensors.json"
    path.write_text(
        json.dumps(
            {
                "sensors": {"truck-7_forwardRight": {"xyz": [0, 0, 0], "rpyDegrees": [-90, 0, 0]}},
                "episodes": {"maxGapSec": 2.5, "maxFrames": 100},
            }
        )
    )
    config = SensorsConfig.load(str(path))
    assert config.max_gap_sec == 2.5 and config.max_frames == 100
    assert config.matrix_for("truck-7_forwardRight", "x") == pytest.approx(ROLL_MINUS_90)
    assert config.matrix_for("other", "x") is None
    # a flat mapping of sensors is read the same way
    path.write_text(json.dumps({"recording": ROLL_MINUS_90}))
    assert SensorsConfig.load(str(path)).matrix_for("recording", "recording") == ROLL_MINUS_90
    path.write_text(json.dumps({"sensors": {"a": [3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}}))
    with pytest.raises(ValueError, match="sensor 'a'"):
        SensorsConfig.load(str(path))


# ---- detection ----


def _capture_set(root, gap_sec=0.0):
    """Two back-to-back captures of one sensor, a later one, and the sensor's mount."""
    root.mkdir(parents=True, exist_ok=True)
    falcon_capture(root / "t7_front_falcon_2026-01-01--00-00-00.pcap", [1, 1, 2, 3])
    falcon_capture(
        root / "t7_front_falcon_2026-01-01--00-01-00.pcap", [3, 4, 5], start=1_700_000_000.04 + gap_sec
    )
    falcon_capture(root / "t7_front_falcon_2026-01-01--00-05-00.pcap", [9], start=1_700_000_300.0)
    (root / "sensors.json").write_text(json.dumps({"t7_front": ROLL_MINUS_90}))
    (root / "t7--2026-01-01--00-00-00.log").write_bytes(b"companion log")
    return root


def test_point_clouds_detection(tmp_path):
    converter = SeyondPcapConverter(str(_capture_set(tmp_path / "in")), None, False)
    assert converter.validate_format()
    assert [i.frame_count for i in converter.get_items()] == [3, 3, 1]
    assert all(i.matrix == ROLL_MINUS_90 for i in converter.get_items())
    assert converter.get_items()[0].source_file == "t7_front_falcon_2026-01-01--00-00-00.pcap"
    groups = converter._group_items()
    assert [len(items) for _, items in groups] == [1, 1, 1]  # a dataset per capture


@pytest.mark.parametrize("gap_sec, max_frames, episodes", [(0.0, None, [2, 1]), (5.0, None, [1, 1, 1]), (0.0, 4, [1, 1, 1])])
def test_episodes_chain_captures(tmp_path, gap_sec, max_frames, episodes):
    root = _capture_set(tmp_path / "in", gap_sec)
    if max_frames:
        (root / "sensors.json").write_text(json.dumps({"episodes": {"maxFrames": max_frames}}))
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    groups = converter._group_items()
    assert [len(items) for _, items in groups] == episodes
    assert len({name for name, _ in groups}) == len(groups)


def test_only_unsupported_captures_fail_with_the_reason(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    records = [ethernet(ipv4_udp(seyond_packet(i, item_type=7, lidar_type=1))) for i in (1, 2)]
    capture(root / "robin.pcap", records)
    with pytest.raises(RuntimeError, match="Robin W"):
        SeyondPcapConverter(str(root), None, False).validate_format()
    # next to a supported capture, the unsupported one is skipped
    falcon_capture(root / "falcon.pcap", [1, 2])
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    assert [os.path.basename(i.path) for i in converter.get_items()] == ["falcon.pcap"]


def test_not_seyond_or_a_supervisely_project_is_left_to_other_converters(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    capture(root / "dns.pcap", [ethernet(ipv4_udp(b"\x12\x34" + bytes(80)))])
    assert not SeyondPcapConverter(str(root), None, False).validate_format()
    project = _capture_set(tmp_path / "project")
    (project / "meta.json").write_text(json.dumps({"classes": [], "tags": []}))
    assert not SeyondPcapEpisodesConverter(str(project), None, False).validate_format()


def test_generic_converter_never_takes_a_pcap_as_one_point_cloud(tmp_path):
    root = _capture_set(tmp_path / "in")
    converter = PointcloudConverter(str(root), None, False)
    items, _, unsupported = converter._collect_items_if_format_not_detected()
    assert items == [] and ".pcap" in unsupported


def test_a_meta_json_that_is_not_a_project_meta_does_not_hide_the_captures(tmp_path):
    root = _capture_set(tmp_path / "in")
    (root / "meta.json").write_text(json.dumps({"tags": []}))
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format() and converter.get_items()


def test_semantic_labels_converter_leaves_pcap_files_alone(tmp_path):
    from supervisely.convert.pointcloud.pcd_semantic_labels.pcd_semantic_labels_converter import (
        PCDSemanticLabelsConverter,
    )

    root = tmp_path / "in"
    root.mkdir()
    (root / "class_mapping.json").write_text(json.dumps({"1": "car"}))
    (root / "0001.pcd").write_text(
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z label\nSIZE 4 4 4 4\nTYPE F F F U\n"
        "COUNT 1 1 1 1\nWIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA ascii\n0 0 0 1\n"
    )
    falcon_capture(root / "a.pcap", [1, 2])
    capture(root / "dns.pcap", [ethernet(ipv4_udp(b"\x12\x34" + bytes(80)))])
    converter = PCDSemanticLabelsConverter(str(root), None, False)
    assert converter.validate_format()
    assert [os.path.basename(item.path) for item in converter.get_items()] == ["0001.pcd"]


def test_upload_dir_leaves_pcap_files_alone(tmp_path):
    from supervisely.api.pointcloud.pointcloud_api import PointcloudApi

    root = tmp_path / "in"
    (root / "sub").mkdir(parents=True)
    falcon_capture(root / "a.pcap", [1])
    falcon_capture(root / "sub" / "B.PCAP", [1])
    (root / "0001.pcd").write_text("x")
    (root / "sub" / "0002.ply").write_text("x")
    uploaded = {}

    def upload_paths(dataset_id, names, paths, progress_cb=None):
        uploaded[dataset_id] = sorted(names)
        return []

    api = types.SimpleNamespace(upload_paths=upload_paths, get_free_names=lambda _, names: names)
    PointcloudApi.upload_dir(api, 1, str(root), recursive=True)
    PointcloudApi.upload_dir(api, 2, str(root), recursive=False)
    assert uploaded == {1: ["0001.pcd", "0002.ply"], 2: ["0001.pcd"]}


def test_remote_folders_of_captures_are_linked():
    manager = types.SimpleNamespace(
        _modality=ProjectType.POINT_CLOUD_EPISODES,
        _NOT_PCAP_POINTCLOUD_EXTS=ImportManager._NOT_PCAP_POINTCLOUD_EXTS,
    )
    is_links = lambda paths: ImportManager._is_pcap_links_input(manager, paths)
    assert is_links(["s3://b/a.pcap", "s3://b/sensors.json"])
    assert not is_links(["s3://b/a.pcap", "s3://b/0001.pcd"])
    assert not is_links(["s3://b/0001.pcd"])
    manager._modality = ProjectType.IMAGES
    assert not is_links(["s3://b/a.pcap"])


# ---- download ----


def test_a_capture_is_downloaded_once_for_its_frames(tmp_path):
    calls = []

    def download(item_id, path):
        calls.append(item_id)
        with open(path, "wb") as f:
            f.write(b"capture bytes")

    downloaded = {}
    info = lambda i, name, h: types.SimpleNamespace(id=i, name=name, hash=h, link=None)
    for i in range(3):
        _download_pointcloud_file(download, info(i, f"c_{i:04d}.pcap", "H"), str(tmp_path / f"{i}.pcap"), downloaded)
    _download_pointcloud_file(download, info(10, "a.pcd", "P"), str(tmp_path / "a.pcd"), downloaded)
    _download_pointcloud_file(download, info(11, "b.pcd", "P"), str(tmp_path / "b.pcd"), downloaded)
    assert calls == [0, 10, 11]  # .pcd items keep one download each
    assert all((tmp_path / f"{i}.pcap").read_bytes() == b"capture bytes" for i in range(3))


def test_renamed_tags_reach_object_tags_without_a_renamed_class():
    from supervisely.convert.pointcloud.sly.sly_pointcloud_helper import rename_in_json
    from supervisely.convert.pointcloud_episodes.sly.sly_pointcloud_episodes_helper import (
        rename_in_json as rename_in_episode_json,
    )

    for rename in (rename_in_json, rename_in_episode_json):
        ann = {"tags": [{"name": "state"}], "objects": [{"classTitle": "car", "tags": [{"name": "state"}]}]}
        ann = rename(ann, {}, {"state": "state_1"})
        assert ann["objects"][0]["tags"][0]["name"] == "state_1"
        assert ann["tags"][0]["name"] == "state_1"
        assert rename({"tags": [], "objects": [{"classTitle": "car", "tags": []}]}, {"car": "car_1"}, None)[
            "objects"
        ][0]["classTitle"] == "car_1"


# ---- review findings ----


@pytest.fixture
def warnings(monkeypatch):
    from supervisely.convert.pointcloud.seyond_pcap import seyond_pcap_converter

    messages = []
    monkeypatch.setattr(seyond_pcap_converter.logger, "warning", lambda msg, *a, **k: messages.append(msg))
    return messages


def test_files_left_out_are_named(tmp_path, warnings):
    root = _capture_set(tmp_path / "in")
    (root / "scan.pcd").write_bytes(b"# .PCD v0.7")
    capture(root / "dns.pcap", [ethernet(ipv4_udp(b"\x12\x34" + bytes(80)))])
    (root / "pcapng.pcap").write_bytes(b"\x0a\x0d\x0d\x0a" + bytes(40))
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    text = "\n".join(warnings)
    assert "scan.pcd" in text and "not imported" in text
    assert "'dns.pcap' is not a Seyond lidar capture" in text
    assert "pcapng" in text


def test_the_generic_converters_do_not_offer_pcap():
    from supervisely.convert.pointcloud_episodes.pointcloud_episodes_converter import (
        PointcloudEpisodeConverter,
    )

    assert ".pcap" not in PointcloudConverter.allowed_exts
    assert ".pcap" not in PointcloudEpisodeConverter.allowed_exts


def test_a_json_named_meta_is_not_a_project(tmp_path):
    root = _capture_set(tmp_path / "in")
    (root / "meta.json").write_text(json.dumps({"vehicle": "t7"}))
    assert SeyondPcapConverter(str(root), None, False).validate_format()


def test_overlapping_captures_never_share_an_episode(tmp_path):
    # the same capture twice, in two folders: one sensor, the same times
    root = tmp_path / "in"
    for folder in ("a", "b"):
        (root / folder).mkdir(parents=True)
        falcon_capture(root / folder / "t7_front_falcon_2026-01-01--00-00-00.pcap", [1, 2, 3])
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    assert [len(items) for _, items in converter._group_items()] == [1, 1]


def test_supervisely_format_with_captures_is_refused(tmp_path):
    from supervisely.convert.pointcloud.sly.sly_pointcloud_converter import (
        SLYPointcloudConverter,
    )

    project = tmp_path / "project"
    (project / "ds" / "pointcloud").mkdir(parents=True)
    (project / "ds" / "ann").mkdir()
    (project / "meta.json").write_text(json.dumps({"classes": [], "tags": []}))
    falcon_capture(project / "ds" / "pointcloud" / "c_0000.pcap", [1])
    (project / "ds" / "ann" / "c_0000.pcap.json").write_text(
        json.dumps({"description": "", "key": "0" * 32, "tags": [], "objects": [], "figures": []})
    )
    with pytest.raises(ValueError, match="Export Seyond lidar pcap"):
        SLYPointcloudConverter(str(project), None, False).validate_format()
    assert not SeyondPcapConverter(str(project), None, False).validate_format()


class _FakeApi:
    """Just enough of the API for upload_dataset, failing on the n-th capture."""

    def __init__(self, fail_on):
        self.fail_on, self.calls, self.datasets, self.removed = fail_on, 0, [], []
        api = self
        info = lambda i: types.SimpleNamespace(id=i, name=f"d{i}", project_id=1, parent_id=None)
        self.dataset = types.SimpleNamespace(
            get_info_by_id=lambda i: info(i),
            get_list=lambda *a, **k: [],
            create=lambda project_id, name, **k: api.datasets.append(name) or info(100 + len(api.datasets)),
            remove=lambda i: api.removed.append(("dataset", i)),
        )
        self.pointcloud = types.SimpleNamespace(
            get_list=lambda *a, **k: [],
            remove_batch=lambda ids: api.removed.append(("items", list(ids))),
            _upload_data_bulk=lambda *a, **k: ["H"],
            _convert_json_info=lambda d: types.SimpleNamespace(**d),
        )
        self.pointcloud_episode = types.SimpleNamespace(
            annotation=types.SimpleNamespace(append=lambda *a, **k: None)
        )
        self.image = types.SimpleNamespace(remove_batch=lambda ids: api.removed.append(("items", list(ids))))

    def post(self, method, data):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("network down")
        entries = data["pointClouds"]
        return types.SimpleNamespace(
            json=lambda: [{"id": self.calls * 1000 + i, "name": e["name"]} for i, e in enumerate(entries)]
        )


def test_a_failure_partway_removes_only_what_it_was_adding(tmp_path, monkeypatch):
    root = _capture_set(tmp_path / "in")
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    monkeypatch.setattr(converter, "merge_metas_with_conflicts", lambda api, ds: (None, {}, {}))
    api = _FakeApi(fail_on=2)
    with pytest.raises(RuntimeError, match="network down"):
        converter.upload_dataset(api, 1, log_progress=False)
    # the first capture's dataset stays whole; the second one is removed; the third never made
    assert len(api.datasets) == 2
    assert api.removed == [("dataset", 102)]


def test_an_empty_meta_json_is_a_project_only_when_linked(tmp_path):
    root = _capture_set(tmp_path / "in")
    (root / "meta.json").write_bytes(b"")
    assert SeyondPcapConverter(str(root), None, False).validate_format()


def test_the_same_one_frame_capture_twice_never_chains(tmp_path):
    root = tmp_path / "in"
    for folder in ("a", "b"):
        (root / folder).mkdir(parents=True)
        falcon_capture(root / folder / "t7_front_falcon_2026-01-01--00-00-00.pcap", [1])
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    assert [len(items) for _, items in converter._group_items()] == [1, 1]


def test_supervisely_annotations_with_captures_are_refused_without_meta(tmp_path):
    from supervisely.convert.pointcloud.sly.sly_pointcloud_converter import (
        SLYPointcloudConverter,
    )

    root = tmp_path / "in"
    (root / "ds" / "pointcloud").mkdir(parents=True)
    (root / "ds" / "ann").mkdir()
    falcon_capture(root / "ds" / "pointcloud" / "c_0000.pcap", [1])
    (root / "ds" / "ann" / "c_0000.pcap.json").write_text(
        json.dumps({"description": "", "key": "0" * 32, "tags": [], "objects": [], "figures": []})
    )
    with pytest.raises(ValueError, match="Export Seyond lidar pcap"):
        SLYPointcloudConverter(str(root), None, False).validate_format()


def test_config_and_junk_are_not_reported_as_left_out(tmp_path, warnings):
    root = _capture_set(tmp_path / "in")
    (root / "sub").mkdir()
    (root / "sub" / "sensors.json").write_text("{}")
    (root / "__MACOSX").mkdir()
    (root / "__MACOSX" / "._sensors.json").write_bytes(b"x")
    (root / "notes.txt").write_text("x")
    assert SeyondPcapConverter(str(root), None, False).validate_format()
    left_out = [w for w in warnings if "not imported (" in w]
    assert len(left_out) == 1 and "sensors.json" not in left_out[0] and "notes.txt" in left_out[0]
    # a vehicle log is where angle tables come from, never a file left out
    assert ".log" not in left_out[0]


def test_items_added_before_a_bad_response_are_removed(tmp_path, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    falcon_capture(root / "only.pcap", [1, 2])
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    monkeypatch.setattr(converter, "merge_metas_with_conflicts", lambda api, ds: (None, {}, {}))
    api = _FakeApi(fail_on=0)

    def broken(d):
        raise KeyError("name")

    api.pointcloud._convert_json_info = broken
    with pytest.raises(KeyError):
        converter.upload_dataset(api, 1, log_progress=False)
    assert api.removed == [("items", [1000, 1001])]


# ---- Robin W ----

from supervisely._utils import get_bytes_hash
from supervisely.pointcloud.seyond_pcap import (
    _find_angle_tables,
    _is_angle_table,
)

SERIAL = "500000000001"
OTHER_SERIAL = "500000000002"


def status_packet(serial, size=128):
    packet = bytearray(size)
    packet[0:2] = struct.pack("<H", 0x186B)
    raw = serial.encode() if isinstance(serial, str) else serial
    packet[38 : 38 + len(raw)] = raw
    return bytes(packet)


def table_packet(seed=0, size=50512):
    """An angle table packet as a Robin W sends it: item type 100, its size in the header."""
    packet = bytearray(size)
    packet[0:2] = struct.pack("<H", 0x176A)
    packet[10:14] = struct.pack("<I", size)
    packet[15] = 1
    packet[38] = 100
    packet[80] = seed
    return bytes(packet)


def log_record(serial, table):
    """What a vehicle log keeps around a table: binary fields, the serial as text, the packet."""
    return b"\x05\x00\x00\x00j\x00\x00\x00\x01\x00\x00\x00" + serial.encode() + b"\x00" * 4 + b"\x01\x00\x00\x00" + table


def robinw_capture(path, frame_indices, serial=SERIAL, **kwargs):
    records = []
    if serial is not None:
        records.append(ethernet(ipv4_udp(status_packet(serial))))
    records += [ethernet(ipv4_udp(seyond_packet(i, item_type=13, lidar_type=1))) for i in frame_indices]
    return capture(path, records, **kwargs)


def test_robinw_frames_and_serial(tmp_path):
    index = index_pcap(robinw_capture(tmp_path / "r.pcap", [5, 5, 6]))
    assert [f.lidar_frame_idx for f in index.frames] == [5, 6]
    assert index.serial == SERIAL and index.needs_angle_table
    assert index.unsupported_reason() is None
    falcon = index_pcap(falcon_capture(tmp_path / "f.pcap", [1]))
    assert falcon.serial is None and not falcon.needs_angle_table


def test_serial_is_the_first_one_the_tool_can_read(tmp_path):
    records = [
        ethernet(ipv4_udp(status_packet(b"\xff\xfe"))),  # not UTF-8
        ethernet(ipv4_udp(status_packet(""))),  # empty
        ethernet(ipv4_udp(status_packet(SERIAL, size=50))),  # too short for the serial field
        ethernet(ipv4_udp(status_packet(OTHER_SERIAL))),
        ethernet(ipv4_udp(status_packet(SERIAL))),
        ethernet(ipv4_udp(seyond_packet(1, item_type=13, lidar_type=1))),
    ]
    index = index_pcap(capture(tmp_path / "r.pcap", records))
    assert index.serial == OTHER_SERIAL
    # status packets are not lidar data: they never count as Seyond data packets
    assert index.seyond_packets == 1


def test_angle_tables_in_a_log(tmp_path):
    first, changed = table_packet(1), table_packet(2)
    log = tmp_path / "v--2026-01-01--00-00-00.log"
    log.write_bytes(
        b"\x00" * 100
        + log_record(SERIAL, first)
        + table_packet(3)  # no text before it: whose it is cannot be told
        + log_record(OTHER_SERIAL, changed)
        + log_record(SERIAL, first)
        + table_packet(4, size=60000)[:1000]  # cut off at the end of the file
    )
    tables = _find_angle_tables(log)
    assert [(s, t) for s, t in tables] == [(SERIAL, first), (OTHER_SERIAL, changed), (SERIAL, first)]
    empty = tmp_path / "empty.log"
    empty.write_bytes(b"")
    assert _find_angle_tables(empty) == []


@pytest.mark.parametrize(
    "data, expected",
    [
        (table_packet(), True),
        # the tool reads the table after the header and nothing else: a trailing newline, or
        # the packet cut right after the table, is still the table
        (table_packet() + b"\n", True),
        (table_packet()[:50000], True),
        (table_packet()[:49999], False),  # too short to hold the table
        (table_packet(size=49000), False),
        (table_packet()[:30000], False),
        (seyond_packet(1), False),
        (b"", False),
    ],
)
def test_a_table_file_is_what_the_tool_reads(data, expected):
    assert _is_angle_table(data) is expected


def _robinw_set(root, log=True):
    root.mkdir(parents=True, exist_ok=True)
    robinw_capture(root / "v_front_robinW_2026-01-01--00-00-00.pcap", [1, 2])
    robinw_capture(root / "v_rear_robinW_2026-01-01--00-00-00.pcap", [1], serial=OTHER_SERIAL)
    falcon_capture(root / "v_left_falcon_2026-01-01--00-00-00.pcap", [1])
    if log:
        (root / "v--2026-01-01--00-00-00.log").write_bytes(
            log_record(SERIAL, table_packet(1)) + log_record(OTHER_SERIAL, table_packet(2))
        )
    return root


def _tables(converter):
    return {os.path.basename(i.path): (i.table.data if i.table else None) for i in converter.get_items()}


def test_tables_from_the_recordings_log(tmp_path, warnings):
    converter = SeyondPcapEpisodesConverter(str(_robinw_set(tmp_path / "in")), None, False)
    assert converter.validate_format()
    assert _tables(converter) == {
        "v_front_robinW_2026-01-01--00-00-00.pcap": table_packet(1),
        "v_left_falcon_2026-01-01--00-00-00.pcap": None,
        "v_rear_robinW_2026-01-01--00-00-00.pcap": table_packet(2),
    }
    assert not [w for w in warnings if "not imported (" in w]


def test_which_table_a_capture_gets(tmp_path, warnings):
    root = tmp_path / "in"
    # an earlier and a later recording of the sensor, in another folder
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "v--2025-12-31--00-00-00.log").write_bytes(log_record(SERIAL, table_packet(5)))
    (root / "logs" / "v--2026-01-02--00-00-00.log").write_bytes(log_record(SERIAL, table_packet(6)))
    robinw_capture(root / "v_front_robinW_2026-01-01--00-00-00.pcap", [1])
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    assert _tables(converter)["v_front_robinW_2026-01-01--00-00-00.pcap"] == table_packet(5)
    # a table file next to the capture comes before any other recording's log
    (root / f"{SERIAL}.anglehv").write_bytes(table_packet(7))
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    assert _tables(converter)["v_front_robinW_2026-01-01--00-00-00.pcap"] == table_packet(7)
    # and the recording's own log comes first; a second, different copy in it is named
    (root / "v--2026-01-01--00-00-00.log").write_bytes(
        log_record(SERIAL, table_packet(8)) + log_record(SERIAL, table_packet(9))
    )
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    assert _tables(converter)["v_front_robinW_2026-01-01--00-00-00.pcap"] == table_packet(8)
    assert any("more than one angle table" in w for w in warnings)


def test_a_bad_table_file_is_not_used(tmp_path, warnings):
    root = _robinw_set(tmp_path / "in", log=False)
    (root / f"{SERIAL}.anglehv").write_bytes(b"not a table")
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    assert _tables(converter)["v_front_robinW_2026-01-01--00-00-00.pcap"] is None
    assert any("is not a Robin W angle table" in w for w in warnings)


def test_a_robinw_capture_without_a_serial_is_skipped(tmp_path, warnings):
    root = tmp_path / "in"
    root.mkdir()
    robinw_capture(root / "r.pcap", [1], serial=None)
    with pytest.raises(RuntimeError, match="serial number"):
        SeyondPcapConverter(str(root), None, False).validate_format()


def test_an_episode_breaks_where_the_table_changes(tmp_path):
    root = tmp_path / "in"
    for minute, seed in ((0, 1), (1, 2)):
        folder = root / f"r{minute}"
        folder.mkdir(parents=True)
        robinw_capture(
            folder / f"v_front_robinW_2026-01-01--00-0{minute}-00.pcap", [1, 2],
            start=1_700_000_000.0 + minute * 0.04,
        )
        (folder / f"v--2026-01-01--00-0{minute}-00.log").write_bytes(log_record(SERIAL, table_packet(seed)))
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    assert [len(items) for _, items in converter._group_items()] == [1, 1]
    # the same table: one episode
    (root / "r1" / "v--2026-01-01--00-01-00.log").write_bytes(log_record(SERIAL, table_packet(1)))
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    assert [len(items) for _, items in converter._group_items()] == [2]


class _TablesApi(_FakeApi):
    """The fake API, with attachments: listed, added by Team Files id or by hash, removed."""

    def __init__(self, fail_on=0, kept=(), stored=()):
        super().__init__(fail_on)
        self.attachments = [dict(a) for a in kept]  # {"datasetId", "projectId", "meta", "hash", "createdAt"}
        self.stored = set(stored)
        self.team_files, self.added = {}, []
        api = self
        self.pointcloud.get_list_all_pages = lambda method, data, convert_json_info_cb=None: [
            a for a in api.attachments
            if all(a.get(k) == v for k, v in data.items() if k in ("datasetId", "projectId"))
        ]
        self.pointcloud.check_existing_hashes = lambda hashes: [h for h in hashes if h in api.stored]
        self.project = types.SimpleNamespace(
            get_info_by_id=lambda i: types.SimpleNamespace(id=i, team_id=7),
            get_list=lambda ws: [types.SimpleNamespace(id=2, type="point_cloud_episodes")],
        )
        self.workspace = types.SimpleNamespace(get_list=lambda team: [types.SimpleNamespace(id=3)])

        def upload(team_id, local, remote):
            with open(local, "rb") as f:
                api.team_files[remote] = f.read()
            return types.SimpleNamespace(id=len(api.team_files), team_id=team_id, path=remote)

        self.file = types.SimpleNamespace(upload=upload, remove=lambda t, p: api.team_files.pop(p))

    def post(self, method, data):
        if method != "entities.bulk.add":
            return super().post(method, data)
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("network down")
        result = []
        for entity in data["entities"]:
            if "teamFileId" in entity:
                data_bytes = list(self.team_files.values())[entity["teamFileId"] - 1]
                entity = {**entity, "hash": get_bytes_hash(data_bytes)}
            self.stored.add(entity["hash"])
            row = {**entity, "id": 9000 + len(self.attachments), "datasetId": data["datasetId"], "projectId": 1}
            self.attachments.append(row)
            self.added.append(row)
            result.append(row)
        return types.SimpleNamespace(json=lambda: result)


def _upload(converter, api, monkeypatch, dataset_id=1):
    from supervisely import ProjectMeta

    meta = ProjectMeta()
    monkeypatch.setattr(converter, "merge_metas_with_conflicts", lambda api, ds: (meta, {}, {}))
    converter.upload_dataset(api, dataset_id, log_progress=False)


def test_each_dataset_keeps_its_sensors_table(tmp_path, monkeypatch):
    converter = SeyondPcapConverter(str(_robinw_set(tmp_path / "in")), None, False)
    assert converter.validate_format()
    api = _TablesApi()
    _upload(converter, api, monkeypatch)
    # a dataset per capture; the Falcon one keeps no table
    by_dataset = {a["datasetId"]: (a["meta"]["serial"], a["hash"]) for a in api.added}
    assert sorted(by_dataset.values()) == sorted(
        [(OTHER_SERIAL, get_bytes_hash(table_packet(2))), (SERIAL, get_bytes_hash(table_packet(1)))]
    )
    assert all(a["type"] == "attachment" and a["title"] == f"anglehv_{a['meta']['serial']}.bin" for a in api.added)
    assert all(a["meta"]["kind"] == "anglehvTable" for a in api.added)
    assert api.team_files == {}  # the uploads are removed from Team Files


def test_a_table_the_platform_stores_is_added_by_hash(tmp_path, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    robinw_capture(root / "v_front_robinW_2026-01-01--00-00-00.pcap", [1])
    robinw_capture(root / "v_front_robinW_2026-01-01--00-05-00.pcap", [1], start=1_700_000_300.0)
    (root / "v--2026-01-01--00-00-00.log").write_bytes(log_record(SERIAL, table_packet(1)))
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    uploads = []
    api = _TablesApi()
    upload = api.file.upload
    api.file.upload = lambda *a: uploads.append(a) or upload(*a)
    _upload(converter, api, monkeypatch)
    # two episodes, one upload: the second dataset adds the stored table by its hash
    assert len(api.added) == 2 and len(uploads) == 1
    assert len({a["hash"] for a in api.added}) == 1


def test_a_missing_table_comes_from_the_latest_one_kept(tmp_path, monkeypatch, warnings):
    root = _robinw_set(tmp_path / "in", log=False)
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    kept = [
        {"datasetId": 5, "projectId": 2, "meta": {"kind": "anglehvTable", "serial": SERIAL}, "hash": "OLD", "createdAt": "2026-01-01T00:00:00.000Z"},
        {"datasetId": 6, "projectId": 2, "meta": {"kind": "anglehvTable", "serial": SERIAL}, "hash": "NEW", "createdAt": "2026-02-01T00:00:00.000Z"},
        {"datasetId": 7, "projectId": 2, "meta": {"kind": "other", "serial": OTHER_SERIAL}, "hash": "X", "createdAt": "2026-03-01T00:00:00.000Z"},
    ]
    api = _TablesApi(kept=kept, stored={"OLD", "NEW"})
    _upload(converter, api, monkeypatch)
    assert [(a["meta"]["serial"], a["hash"]) for a in api.added] == [(SERIAL, "NEW")]
    # the copy keeps the time of the table it copies: it never passes for a newer one
    assert api.added[0]["createdAt"] == "2026-02-01T00:00:00.000Z"
    # the rear sensor has no table anywhere: its capture is not imported, the others are
    assert any("no angle table for Robin W sensor 500000000002" in w for w in warnings)
    assert sorted(os.path.basename(i.path) for i in converter.get_items()) == [
        "v_front_robinW_2026-01-01--00-00-00.pcap",
        "v_left_falcon_2026-01-01--00-00-00.pcap",
    ]


def test_no_table_anywhere_fails_the_import(tmp_path, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    robinw_capture(root / "v_front_robinW_2026-01-01--00-00-00.pcap", [1])
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    with pytest.raises(RuntimeError, match="angle tables are missing"):
        _upload(converter, _TablesApi(), monkeypatch)


def test_a_target_with_another_table_for_the_sensor_gets_a_nested_dataset(tmp_path, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    robinw_capture(root / "v_front_robinW_2026-01-01--00-00-00.pcap", [1])
    (root / "v--2026-01-01--00-00-00.log").write_bytes(log_record(SERIAL, table_packet(1)))
    kept = [{"datasetId": 1, "projectId": 1, "meta": {"kind": "anglehvTable", "serial": SERIAL}, "hash": "OTHER", "createdAt": "x"}]
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    api = _TablesApi(kept=kept)
    _upload(converter, api, monkeypatch)
    assert len(api.datasets) == 1 and api.added[0]["datasetId"] == 101
    # the same table already there: the target dataset, and nothing added twice
    kept[0]["hash"] = get_bytes_hash(table_packet(1))
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    api = _TablesApi(kept=kept)
    _upload(converter, api, monkeypatch)
    assert api.datasets == [] and api.added == []


def test_a_failure_after_the_table_keeps_it_in_the_target(tmp_path, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    robinw_capture(root / "v_front_robinW_2026-01-01--00-00-00.pcap", [1])
    (root / "v--2026-01-01--00-00-00.log").write_bytes(log_record(SERIAL, table_packet(1)))
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    api = _TablesApi(fail_on=2)  # the table is added, the frames fail
    with pytest.raises(RuntimeError, match="network down"):
        _upload(converter, api, monkeypatch)
    # removing an attachment counts it off the dataset's items, which it never was in
    assert len(api.added) == 1 and api.removed == []
    # a new import into the same dataset reuses the table instead of adding a second one
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    api.fail_on, api.calls = 0, 0
    _upload(converter, api, monkeypatch)
    assert len(api.added) == 1


def test_the_serial_before_a_table_is_one_a_capture_has(tmp_path):
    # a log that keeps other text between the serial and the table
    log = tmp_path / "v--2026-01-01--00-00-00.log"
    log.write_bytes(
        b"\x01\x00\x00\x00" + SERIAL.encode() + b"\x00\x02\x00\x00\x00hostname-a\x00\x01\x00\x00\x00" + table_packet(1)
    )
    assert _find_angle_tables(log) == [("hostname-a", table_packet(1))]
    assert _find_angle_tables(log, {SERIAL}) == [(SERIAL, table_packet(1))]
    assert _find_angle_tables(log, {OTHER_SERIAL}) == []
    root = tmp_path / "in"
    root.mkdir()
    log.rename(root / log.name)
    robinw_capture(root / "v_front_robinW_2026-01-01--00-00-00.pcap", [1])
    converter = SeyondPcapConverter(str(root), None, False)
    assert converter.validate_format()
    assert _tables(converter)["v_front_robinW_2026-01-01--00-00-00.pcap"] == table_packet(1)


def test_other_logs_are_read_only_for_sensors_still_without_a_table(tmp_path, monkeypatch):
    root = _robinw_set(tmp_path / "in")
    (root / "older").mkdir()
    (root / "older" / "v--2025-12-31--00-00-00.log").write_bytes(log_record(SERIAL, table_packet(5)))
    (root / "older" / "w--2025-12-31--00-00-00.log").write_bytes(b"a Falcon-only recording")
    read = []
    real = SeyondPcapConverter._read_tables
    monkeypatch.setattr(
        SeyondPcapConverter, "_read_tables", lambda self, p, s: read.append(os.path.basename(p)) or real(self, p, s)
    )
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    assert read == ["v--2026-01-01--00-00-00.log"]
    # without the recording's own log, the others are read for the sensors it would have given
    (root / "v--2026-01-01--00-00-00.log").unlink()
    read.clear()
    converter = SeyondPcapEpisodesConverter(str(root), None, False)
    assert converter.validate_format()
    assert sorted(read) == ["v--2025-12-31--00-00-00.log", "w--2025-12-31--00-00-00.log"]
    assert _tables(converter)["v_front_robinW_2026-01-01--00-00-00.pcap"] == table_packet(5)


def test_another_sensors_serial_is_never_skipped_over(tmp_path):
    other = "500000000009"  # a sensor that is not in the upload
    log = tmp_path / "v--2026-01-01--00-00-00.log"
    log.write_bytes(
        b"status " + SERIAL.encode() + b"\x00ok\x00" + log_record(other, table_packet(9))
        + bytes(1000) + log_record(SERIAL, table_packet(1))
    )
    assert _find_angle_tables(log, {SERIAL}) == [(SERIAL, table_packet(1))]

