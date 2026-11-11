# coding: utf-8
"""Seyond (Innovusion) lidar captures stored as raw libpcap files.

A capture is the sensor's own UDP output recorded with tcpdump, so one file holds a whole
recording: many point cloud frames. Supervisely stores such a file once and adds one item per
frame that reuses it. The labeling tool decodes the frame it needs from the file itself.

This module only indexes a capture: which frames it holds, in which order, and when each one
starts. It must find exactly the frames the labeling tool finds, in the same order (the tool's
``PcapIndex::build``), because an item's frame in its file is its position in that order:

- classic pcap only (microsecond or nanosecond timestamps, either byte order), not pcapng;
- link types Ethernet (with VLAN / QinQ tags), raw IP, Linux cooked v1 and v2;
- IPv4 or IPv6 carrying UDP; a fragmented IPv4 datagram is skipped, not reassembled;
- a lidar data packet starts with the magic ``0x176a`` and carries its frame index at byte 26;
  only Falcon sphere point cloud packets (item type 1) make frames;
- frames are numbered in the order their first packet appears in the file, so a capture that
  starts or ends mid-frame keeps those partial frames.

The packet layout follows Seyond's inno-lidar-sdk (BSD-2-Clause),
``src/sdk_common/inno_lidar_packet.h``. Only the standard library is used.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import BinaryIO, Dict, List, Optional, Union

PCAP_EXT = ".pcap"

_PCAP_GLOBAL_HEADER_SIZE = 24
_PCAP_RECORD_HEADER_SIZE = 16
_PCAP_MAGIC_MICROSECONDS = 0xA1B2C3D4
_PCAP_MAGIC_NANOSECONDS = 0xA1B23C4D
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

_LINKTYPE_ETHERNET = 1
_LINKTYPE_RAW = 101
_LINKTYPE_LINUX_SLL = 113
_LINKTYPE_LINUX_SLL2 = 276

_ETHERNET_HEADER_SIZE = 14
_VLAN_TAG_SIZE = 4
_LINUX_SLL_HEADER_SIZE = 16
_LINUX_SLL2_HEADER_SIZE = 20
_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_IPV6 = 0x86DD
_ETHERTYPE_VLAN = 0x8100
_ETHERTYPE_QINQ = 0x88A8
_IPV6_HEADER_SIZE = 40
_IP_PROTOCOL_UDP = 17
_UDP_HEADER_SIZE = 8

INNO_DATA_PACKET_MAGIC = 0x176A
# InnoDataPacket up to its payload; protocol 1 has no extend_reserved[4]
_INNO_DATA_HEADER_SIZE_V1 = 54
_INNO_DATA_HEADER_SIZE = 70
_INNO_LIDAR_TYPE_OFFSET = 15
_INNO_FRAME_IDX_OFFSET = 26
_INNO_ITEM_TYPE_OFFSET = 38

#: The item type of the packets the labeling tool decodes: Falcon sphere point clouds.
INNO_ITEM_TYPE_SPHERE_POINTCLOUD = 1

#: ``lidar_type`` in the packet header (inno-lidar-sdk ``InnoLidarType``).
INNO_LIDAR_TYPES = {
    0: "Falcon K1",
    1: "Robin W",
    2: "Robin E",
    3: "Falcon K2 / Falcon 2.1",
    4: "Falcon III",
    5: "Robin E-Lite",
    6: "Robin E2X",
    7: "Hummingbird",
    8: "Robin E2",
}

_U16_BE = struct.Struct(">H")
_U16_LE = struct.Struct("<H")
_U64_LE = struct.Struct("<Q")


class SeyondPcapError(Exception):
    """Raised when a file cannot be indexed as a Seyond lidar pcap capture."""


@dataclass
class SeyondPcapFrame:
    """One lidar frame of a capture."""

    #: The frame's position in the file: the order in which frames start. This is what the
    #: labeling tool calls the frame index, and what ``meta.source.frameInFile`` holds.
    index: int
    #: The frame index the lidar wrote into its packets.
    lidar_frame_idx: int
    #: Capture (host) time of the frame's first packet, seconds since the epoch, UTC.
    time: float
    #: Number of point cloud packets in the frame.
    packets: int = 0

    @property
    def timestamp(self) -> str:
        """ISO 8601 UTC time of the frame's first packet, in milliseconds: ``meta.timestamp``."""
        return to_iso_timestamp(self.time)


@dataclass
class SeyondPcapIndex:
    """What a capture holds."""

    frames: List[SeyondPcapFrame] = field(default_factory=list)
    #: Packets with the Seyond data magic, decodable or not, including the first fragment of a
    #: fragmented datagram.
    seyond_packets: int = 0
    #: Item types seen in Seyond data packets, with their packet counts.
    item_types: Dict[int, int] = field(default_factory=dict)
    #: ``lidar_type`` values seen in Seyond data packets, with their packet counts.
    lidar_types: Dict[int, int] = field(default_factory=dict)
    #: Seyond data packets the labeling tool skips because their IPv4 datagram is fragmented.
    fragmented_packets: int = 0
    #: True when the file ends inside a record (the packets before it are kept).
    truncated: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def is_seyond(self) -> bool:
        """The file holds Seyond lidar data packets, whether or not they can be decoded."""
        return self.seyond_packets > 0

    @property
    def model(self) -> Optional[str]:
        """The lidar model the packets name most often, or None."""
        if not self.lidar_types:
            return None
        lidar_type = max(self.lidar_types, key=self.lidar_types.get)
        return INNO_LIDAR_TYPES.get(lidar_type, f"lidar type {lidar_type}")

    def unsupported_reason(self) -> Optional[str]:
        """Why the labeling tool cannot show this capture, or None when it can."""
        if self.frames:
            return None
        if not self.is_seyond:
            return "no Seyond lidar data packets found"
        details = []
        if self.model is not None:
            details.append(self.model)
        if self.item_types:
            types = ", ".join(str(t) for t in sorted(self.item_types))
            details.append(f"packet item type {types}")
        if self.fragmented_packets:
            details.append(f"{self.fragmented_packets} fragmented IP datagrams")
        return (
            "only Seyond Falcon sphere point cloud packets (item type "
            f"{INNO_ITEM_TYPE_SPHERE_POINTCLOUD}) are supported, this capture has "
            + ", ".join(details)
        )


def to_iso_timestamp(seconds: float) -> str:
    """Seconds since the epoch as ``YYYY-MM-DDTHH:MM:SS.mmmZ`` (UTC)."""
    moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def is_pcap_header(header: bytes) -> bool:
    """True if ``header`` (at least the first 4 bytes of a file) starts a classic pcap file."""
    return _byte_order(header) is not None


def _byte_order(header: bytes) -> Optional[str]:
    if len(header) < 4:
        return None
    for order in ("<", ">"):
        magic = struct.unpack(order + "I", header[:4])[0]
        if magic in (_PCAP_MAGIC_MICROSECONDS, _PCAP_MAGIC_NANOSECONDS):
            return order
    return None


def _udp_payload(frame: bytes, link_type: int):
    """(start, end) of the UDP payload in a pcap record, or None; mirrors the labeling tool.

    The second value tells a fragmented IPv4 datagram apart: ``(None, start)`` when the record is
    the first fragment of one, so a Seyond packet can still be recognised without counting it.
    """
    n = len(frame)
    if link_type == _LINKTYPE_ETHERNET and n >= _ETHERNET_HEADER_SIZE:
        type_offset = _ETHERNET_HEADER_SIZE - 2
        ether_type = _U16_BE.unpack_from(frame, type_offset)[0]
        while ether_type in (_ETHERTYPE_VLAN, _ETHERTYPE_QINQ):
            type_offset += _VLAN_TAG_SIZE
            if type_offset + 2 > n:
                return None
            ether_type = _U16_BE.unpack_from(frame, type_offset)[0]
        ip_offset = type_offset + 2
    elif link_type == _LINKTYPE_LINUX_SLL and n >= _LINUX_SLL_HEADER_SIZE:
        ip_offset = _LINUX_SLL_HEADER_SIZE
        ether_type = _U16_BE.unpack_from(frame, _LINUX_SLL_HEADER_SIZE - 2)[0]
    elif link_type == _LINKTYPE_LINUX_SLL2 and n >= _LINUX_SLL2_HEADER_SIZE:
        ip_offset = _LINUX_SLL2_HEADER_SIZE
        ether_type = _U16_BE.unpack_from(frame, 0)[0]
    elif link_type == _LINKTYPE_RAW:
        ip_offset = 0
        ether_type = None
    else:
        return None

    if ip_offset >= n:
        return None
    ip_len = n - ip_offset
    version = frame[ip_offset] >> 4
    fragmented = False
    if version == 4 and ether_type in (None, _ETHERTYPE_IPV4):
        header_len = (frame[ip_offset] & 0x0F) * 4
        if ip_len < 20 or header_len < 20 or frame[ip_offset + 9] != _IP_PROTOCOL_UDP:
            return None
        flags_fragment = _U16_BE.unpack_from(frame, ip_offset + 6)[0]
        if flags_fragment & 0x3FFF != 0:
            # a fragment: only the first one (offset 0) carries the UDP header
            if flags_fragment & 0x1FFF != 0:
                return None
            fragmented = True
        udp_offset = header_len
        ip_end = min(_U16_BE.unpack_from(frame, ip_offset + 2)[0], ip_len)
        if fragmented:
            ip_end = ip_len
    elif version == 6 and ether_type in (None, _ETHERTYPE_IPV6):
        if ip_len < _IPV6_HEADER_SIZE or frame[ip_offset + 6] != _IP_PROTOCOL_UDP:
            return None
        udp_offset = _IPV6_HEADER_SIZE
        ip_end = min(_IPV6_HEADER_SIZE + _U16_BE.unpack_from(frame, ip_offset + 4)[0], ip_len)
    else:
        return None

    if udp_offset > ip_end or ip_end - udp_offset < _UDP_HEADER_SIZE:
        return None
    udp_start = ip_offset + udp_offset
    start = udp_start + _UDP_HEADER_SIZE
    if fragmented:
        return None, start
    udp_len = _U16_BE.unpack_from(frame, udp_start + 4)[0]
    udp_len = max(_UDP_HEADER_SIZE, min(udp_len, ip_end - udp_offset))
    return start, udp_start + udp_len


def index_pcap(
    source: Union[str, os.PathLike, BinaryIO],
    max_records: Optional[int] = None,
) -> SeyondPcapIndex:
    """Index the lidar frames of a Seyond pcap capture.

    :param source: Path to the ``.pcap`` file, or a binary file object at its start.
    :param max_records: Stop after this many pcap records (to recognise a file quickly);
        None reads the whole file.
    :returns: The frames in the order they start, plus what was seen on the way.
    :raises SeyondPcapError: If the file is not a classic pcap file (a pcapng file included).
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            return index_pcap(f, max_records)

    f = source
    header = f.read(_PCAP_GLOBAL_HEADER_SIZE)
    if header[:4] == _PCAPNG_MAGIC:
        raise SeyondPcapError("pcapng files are not supported, save the capture as pcap")
    order = _byte_order(header)
    if order is None or len(header) < _PCAP_GLOBAL_HEADER_SIZE:
        raise SeyondPcapError("not a pcap file")
    nanoseconds = struct.unpack(order + "I", header[:4])[0] == _PCAP_MAGIC_NANOSECONDS
    link_type = struct.unpack(order + "I", header[20:24])[0] & 0x0FFFFFFF
    record_header = struct.Struct(order + "IIII")
    fraction = 1e-9 if nanoseconds else 1e-6

    result = SeyondPcapIndex()
    positions: Dict[int, SeyondPcapFrame] = {}
    records = 0
    while max_records is None or records < max_records:
        head = f.read(_PCAP_RECORD_HEADER_SIZE)
        if len(head) < _PCAP_RECORD_HEADER_SIZE:
            break
        ts_sec, ts_frac, captured_len, _ = record_header.unpack(head)
        frame = f.read(captured_len)
        if len(frame) < captured_len:
            # a capture cut off mid-record keeps the packets before it
            result.truncated = True
            break
        records += 1

        payload = _udp_payload(frame, link_type)
        if payload is None:
            continue
        start, end = payload
        if start is None:
            # first fragment of a fragmented datagram: recognised, never decoded
            begin = end
            if (
                len(frame) - begin >= 2
                and _U16_LE.unpack_from(frame, begin)[0] == INNO_DATA_PACKET_MAGIC
            ):
                result.seyond_packets += 1
                result.fragmented_packets += 1
                _count_header(result, frame, begin, len(frame))
            continue

        size = end - start
        if size < 2 or _U16_LE.unpack_from(frame, start)[0] != INNO_DATA_PACKET_MAGIC:
            continue
        result.seyond_packets += 1
        _count_header(result, frame, start, end)
        if size < _INNO_DATA_HEADER_SIZE_V1:
            continue
        header_size = _INNO_DATA_HEADER_SIZE_V1 if frame[start + 2] == 1 else _INNO_DATA_HEADER_SIZE
        if size < header_size or frame[start + _INNO_ITEM_TYPE_OFFSET] != INNO_ITEM_TYPE_SPHERE_POINTCLOUD:
            continue

        lidar_frame_idx = _U64_LE.unpack_from(frame, start + _INNO_FRAME_IDX_OFFSET)[0]
        lidar_frame = positions.get(lidar_frame_idx)
        if lidar_frame is None:
            lidar_frame = SeyondPcapFrame(
                index=len(result.frames),
                lidar_frame_idx=lidar_frame_idx,
                time=ts_sec + ts_frac * fraction,
            )
            positions[lidar_frame_idx] = lidar_frame
            result.frames.append(lidar_frame)
        lidar_frame.packets += 1

    return result


def _count_header(result: SeyondPcapIndex, frame: bytes, start: int, end: int) -> None:
    if end - start > _INNO_ITEM_TYPE_OFFSET:
        item_type = frame[start + _INNO_ITEM_TYPE_OFFSET]
        result.item_types[item_type] = result.item_types.get(item_type, 0) + 1
    if end - start > _INNO_LIDAR_TYPE_OFFSET:
        lidar_type = frame[start + _INNO_LIDAR_TYPE_OFFSET]
        result.lidar_types[lidar_type] = result.lidar_types.get(lidar_type, 0) + 1


def is_seyond_pcap(path: str, max_records: int = 2000) -> bool:
    """True if the file is a classic pcap file whose first records carry Seyond lidar data.

    Reads at most ``max_records`` records, so it is cheap on a large capture. It says nothing
    about whether the labeling tool can decode the capture; see :func:`index_pcap`.
    """
    try:
        return index_pcap(path, max_records=max_records).is_seyond
    except (SeyondPcapError, OSError):
        return False
