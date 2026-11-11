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
        root / "t7_front_falcon_2026-01-01--00-01-00.pcap", [3, 4, 5], start=1_700_000_000.03 + gap_sec
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
