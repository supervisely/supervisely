"""
Helper functions for SemanticKITTI format converter.

Provides utilities for reading, processing and converting SemanticKITTI data:
- Reading .bin point cloud files
- Reading .label semantic/instance annotation files
- Converting .bin to .pcd format with RGB colors
- Auto-generating colors for classes
- Managing class mappings
"""

import numpy as np
import colorsys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from supervisely import logger


def read_bin_pointcloud(bin_path: str) -> np.ndarray:
    """
    Read point cloud from SemanticKITTI .bin file.

    Args:
        bin_path: Path to .bin file

    Returns:
        np.ndarray: Points array of shape (N, 4) containing [x, y, z, intensity]
    """
    points = np.fromfile(bin_path, dtype=np.float32)
    return points.reshape((-1, 4))


def read_label_file(
    label_path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read semantic and instance labels from SemanticKITTI .label file.

    Format: uint32 where lower 16 bits = semantic_id, upper 16 bits = instance_id

    Args:
        label_path: Path to .label file

    Returns:
        Tuple of (semantic_labels, instance_ids):
            - semantic_labels: Array of semantic class IDs (N,)
            - instance_ids: Array of instance IDs (N,)
            Returns (None, None) if file doesn't exist
    """
    if not Path(label_path).exists():
        return None, None

    labels = np.fromfile(label_path, dtype=np.uint32)

    # Extract semantic and instance from packed format
    semantic_labels = labels & 0xFFFF  # Lower 16 bits
    instance_ids = labels >> 16  # Upper 16 bits

    return semantic_labels, instance_ids


def generate_color_for_class(class_id: int) -> List[int]:
    """
    Generate a stable, distinct color for a class using HSV color space.

    Uses golden angle (137.5°) to maximize color distance between adjacent IDs.
    Same class_id always produces same color (deterministic).

    Args:
        class_id: Class identifier

    Returns:
        List[int]: RGB color as [r, g, b] where each value is 0-255

    Example:
        >>> generate_color_for_class(0)
        [242, 13, 13]  # Red-ish
        >>> generate_color_for_class(1)
        [13, 242, 166]  # Cyan-ish
    """
    # Golden angle in degrees for maximum color separation
    golden_angle = 137.5

    # Generate hue based on class_id
    hue = (class_id * golden_angle) % 360 / 360.0

    # High saturation and value for vibrant colors
    saturation = 0.8
    value = 0.95

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return [int(r * 255), int(g * 255), int(b * 255)]


def convert_bin_to_pcd(bin_path: str, pcd_path: str) -> None:
    """
    Convert SemanticKITTI .bin file to .pcd format.

    Writes raw XYZ + intensity data without any color information.
    Semantic labels are stored in Supervisely annotations (not in the point cloud).

    Args:
        bin_path: Path to input .bin file
        pcd_path: Path to output .pcd file

    Example:
        >>> convert_bin_to_pcd("000000.bin", "000000.pcd")
    """
    points = read_bin_pointcloud(bin_path)  # shape (N, 4): x, y, z, intensity
    _write_pcd_raw(points, pcd_path)


def _write_pcd_raw(points: np.ndarray, filename: str) -> None:
    """
    Write point cloud to PCD file in binary format with x, y, z, intensity.

    Args:
        points: Array of shape (N, 4) containing [x, y, z, intensity]
        filename: Output PCD file path
    """
    num_points = len(points)
    xyz = points[:, :3].astype(np.float32)
    intensity = points[:, 3].astype(np.float32)

    with open(filename, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")

        for i in range(num_points):
            f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {intensity[i]}\n")


def scan_labels_for_classes(
    labels_dir: Path,
    label_map: Optional[Dict[int, Tuple[str, List[int]]]] = None,
) -> Dict[int, Tuple[str, List[int]]]:
    """
    Scan all .label files to find unique semantic classes present in data.

    Creates class definitions for all found classes:
    - Uses label_map for known classes
    - Auto-generates names and colors for unknown classes

    Args:
        labels_dir: Directory containing .label files
        label_map: Optional predefined mapping {class_id: (name, [r,g,b])}
                   If None or empty dict, all classes are auto-generated

    Returns:
        Dict mapping class_id to (name, color):
            {class_id: (name, [r, g, b])}

    Example:
        >>> classes = scan_labels_for_classes(Path("labels"), {
        ...     1: ("car", [255, 0, 0])
        ... })
        >>> classes[1]
        ('car', [255, 0, 0])
        >>> classes[2]  # Auto-generated
        ('class_2', [13, 242, 166])
    """
    if label_map is None:
        label_map = {}

    # Find all unique semantic IDs in the data
    unique_ids = set()
    label_files = sorted(labels_dir.glob("*.label"))

    logger.info(f"Scanning {len(label_files)} label files for classes...")

    for label_file in label_files:
        semantic_labels, _ = read_label_file(str(label_file))
        if semantic_labels is not None:
            unique_ids.update(np.unique(semantic_labels))

    # Build complete class mapping
    complete_map = {}
    auto_generated_count = 0

    for class_id in sorted(unique_ids):
        class_id = int(class_id)
        if class_id in label_map:
            # Use predefined mapping
            complete_map[class_id] = label_map[class_id]
        else:
            # Auto-generate class name and color
            class_name = f"class_{class_id}"
            class_color = generate_color_for_class(class_id)
            complete_map[class_id] = (class_name, class_color)
            auto_generated_count += 1

    logger.info(f"Found {len(unique_ids)} classes:")
    logger.info(f"  - {len(unique_ids) - auto_generated_count} from label_map")
    logger.info(f"  - {auto_generated_count} auto-generated")

    return complete_map


def read_poses_file(poses_path: str) -> np.ndarray:
    """
    Read camera poses from poses.txt file.

    Each line contains 12 values representing a 3x4 transformation matrix.

    Args:
        poses_path: Path to poses.txt file

    Returns:
        np.ndarray: Array of poses, shape (N, 12) or (N, 3, 4)
    """
    if not Path(poses_path).exists():
        logger.warning(f"Poses file not found: {poses_path}")
        return None

    poses = np.loadtxt(poses_path)
    return poses


def read_times_file(times_path: str) -> np.ndarray:
    """
    Read timestamps from times.txt file.

    Each line contains one timestamp value.

    Args:
        times_path: Path to times.txt file

    Returns:
        np.ndarray: Array of timestamps
    """
    if not Path(times_path).exists():
        logger.warning(f"Times file not found: {times_path}")
        return None

    times = np.loadtxt(times_path)
    return times


def validate_sequence_structure(sequence_path: Path) -> Tuple[bool, str]:
    """
    Validate that sequence directory has required SemanticKITTI structure.

    Required:
        - velodyne/ directory with .bin files
        - labels/ directory with .label files (optional)
        - poses.txt file
        - times.txt file

    Args:
        sequence_path: Path to sequence directory

    Returns:
        Tuple of (is_valid, error_message):
            - is_valid: True if structure is valid
            - error_message: Description of validation error, or empty string if valid
    """
    velodyne_dir = sequence_path / "velodyne"
    labels_dir = sequence_path / "labels"
    poses_file = sequence_path / "poses.txt"
    times_file = sequence_path / "times.txt"

    # Check velodyne directory
    if not velodyne_dir.exists():
        return False, f"Missing velodyne/ directory: {velodyne_dir}"

    bin_files = list(velodyne_dir.glob("*.bin"))
    if len(bin_files) == 0:
        return False, f"No .bin files found in {velodyne_dir}"

    # Check poses file
    if not poses_file.exists():
        return False, f"Missing poses.txt file: {poses_file}"

    # Check times file
    if not times_file.exists():
        return False, f"Missing times.txt file: {times_file}"

    # Labels are optional - just log warning if missing
    if not labels_dir.exists():
        logger.warning(f"No labels/ directory found: {labels_dir}")
        logger.warning("Points will not be colored by semantic class")

    return True, ""
