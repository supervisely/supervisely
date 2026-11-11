"""Helpers for importing Seyond lidar pcap captures: capture names, the sensors config, matrices."""

import os
import re
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from supervisely.io.json import load_json_file

#: The optional config that gives each sensor its mount and the episode rules.
SENSORS_CONFIG_NAME = "sensors.json"
#: Where the export of a Seyond pcap project keeps its labels, next to the captures.
LABELS_DIR_NAME = "supervisely_labels"
LABELS_META_NAME = "project_meta.json"
LABELS_FILE_NAME = "labels.json"

#: A new episode starts when the next capture of a sensor begins more than this many seconds
#: after the previous one ends.
DEFAULT_MAX_GAP_SEC = 1.0

# {device}_{location}_{model}_{timestamp}.pcap, e.g. truck-1_forwardRight_falcon_2026-08-26--16-48-00
_CAPTURE_NAME = re.compile(
    r"^(?P<device>.+)_(?P<location>[^_]+)_(?P<model>[^_]+)_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}(?:[.\-]\d+)?)$"
)


def parse_capture_name(file_name: str) -> Optional[Dict[str, str]]:
    """Split ``{device}_{location}_{model}_{timestamp}.pcap``; None for any other name."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    match = _CAPTURE_NAME.match(stem)
    if match is None:
        return None
    return match.groupdict()


def sensor_key(file_name: str) -> str:
    """The sensor a capture comes from: ``{device}_{location}``, or the file stem for other names."""
    parsed = parse_capture_name(file_name)
    if parsed is None:
        return os.path.splitext(os.path.basename(file_name))[0]
    return f"{parsed['device']}_{parsed['location']}"


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix for roll (X), pitch (Y) and yaw (Z) in radians: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def to_extrinsic_matrix(value: Union[Sequence[float], Dict], name: str = "sensor") -> List[float]:
    """The 12 numbers of ``meta.sensorsData.extrinsicMatrix`` (sensor to world, row-major 3x4).

    Accepts 12 numbers, 16 numbers whose last row is ``0 0 0 1``, or
    ``{"xyz": [x, y, z], "rpy": [roll, pitch, yaw]}`` in metres and radians
    (``"rpyDegrees"`` instead of ``"rpy"`` for degrees). The rotation must be rigid: the
    labeling tool refuses scale and skew, and the matrix cannot change once the frames have labels.

    :raises ValueError: If the value is not one of those or is not a rotation plus a translation.
    """
    if isinstance(value, dict):
        if "extrinsicMatrix" in value:
            return to_extrinsic_matrix(value["extrinsicMatrix"], name)
        xyz = value.get("xyz", [0, 0, 0])
        if "rpyDegrees" in value:
            rpy = np.radians(np.asarray(value["rpyDegrees"], dtype=float))
        elif "rpy" in value:
            rpy = np.asarray(value["rpy"], dtype=float)
        else:
            raise ValueError(
                f"{name}: give 'extrinsicMatrix' (12 or 16 numbers) or 'xyz' with 'rpy' (radians) "
                "or 'rpyDegrees'"
            )
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError(f"{name}: 'xyz' and 'rpy' must have 3 numbers each")
        matrix = np.hstack([rotation_from_rpy(*rpy), xyz.reshape(3, 1)])
    else:
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size == 16:
            if not np.allclose(values[12:], [0, 0, 0, 1]):
                raise ValueError(f"{name}: the last row of a 4x4 matrix must be 0 0 0 1")
            values = values[:12]
        if values.size != 12:
            raise ValueError(f"{name}: a matrix needs 12 or 16 numbers, got {values.size}")
        matrix = values.reshape(3, 4)

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}: the matrix has non-finite numbers")
    rotation = matrix[:, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError(
            f"{name}: the matrix must be a rotation plus a translation, without scale, skew or mirroring"
        )
    return [float(round(v, 9)) + 0.0 for v in matrix.reshape(-1)]


class SensorsConfig:
    """``sensors.json``: each sensor's mount, and when an episode ends.

    .. code-block:: json

        {
          "sensors": {
            "truck-1_forwardRight": {"xyz": [1.2, -0.8, 2.1], "rpyDegrees": [-90, 0, 0]},
            "truck-1_forwardLeft": [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0]
          },
          "episodes": {"maxGapSec": 1.0, "maxFrames": 3000}
        }

    A sensor is ``{device}_{location}`` from the capture name, or the file stem of a capture with
    any other name. A mapping of sensors without the ``"sensors"`` key is read the same way.
    """

    def __init__(
        self,
        matrices: Optional[Dict[str, List[float]]] = None,
        max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
        max_frames: Optional[int] = None,
        path: Optional[str] = None,
    ):
        self.matrices = matrices or {}
        self.max_gap_sec = max_gap_sec
        self.max_frames = max_frames
        self.path = path

    @classmethod
    def load(cls, path: Optional[str]) -> "SensorsConfig":
        if path is None:
            return cls()
        data = load_json_file(path)
        if not isinstance(data, dict):
            raise ValueError(f"{SENSORS_CONFIG_NAME} must hold a JSON object")
        episodes = data.get("episodes", {}) or {}
        sensors = data.get("sensors")
        if sensors is None:
            sensors = {k: v for k, v in data.items() if k != "episodes"}
        if not isinstance(sensors, dict):
            raise ValueError(f"{SENSORS_CONFIG_NAME}: 'sensors' must be an object")
        matrices = {}
        for key, value in sensors.items():
            matrices[key] = to_extrinsic_matrix(value, f"{SENSORS_CONFIG_NAME}, sensor '{key}'")
        max_gap = float(episodes.get("maxGapSec", DEFAULT_MAX_GAP_SEC))
        max_frames = episodes.get("maxFrames")
        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames < 1:
                raise ValueError(f"{SENSORS_CONFIG_NAME}: 'episodes.maxFrames' must be at least 1")
        return cls(matrices, max_gap, max_frames, path)

    def matrix_for(self, sensor: str, file_stem: str) -> Optional[List[float]]:
        if sensor in self.matrices:
            return self.matrices[sensor]
        return self.matrices.get(file_stem)


def to_iso_name(seconds: float) -> str:
    """A UTC time for a dataset name: ``2026-08-26 23-47-59``."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H-%M-%S")
