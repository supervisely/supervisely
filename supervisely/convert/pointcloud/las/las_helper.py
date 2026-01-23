import numpy as np

from supervisely import logger


def las2pcd(input_path: str, output_path: str) -> None:
    """
    Convert a LAS/LAZ point cloud to PCD format.

    The function reads a LAS/LAZ file, applies coordinate scaling and offsets,
    recenters the point cloud to improve numerical stability, and writes
    the result to a PCD file compatible with common point cloud viewers.

    :param input_path: Path to the input LAS/LAZ file.
    :type input_path: str
    :param output_path: Path where the output PCD file will be written.
    :type output_path: str
    :return: None
    """
    import laspy  # pylint: disable=import-error
    from pypcd4 import Encoding  # pylint: disable=import-error
    from pypcd4 import PointCloud as pypcd4_pcd  # pylint: disable=import-error

    # Read LAS file
    try:
        las = laspy.read(input_path)
    except Exception as e:
        if "buffer size must be a multiple of element size" in str(e):
            logger.warning(
                "LAS/LAZ file read failed due to buffer size mismatch with EXTRA_BYTES. "
                "Retrying with EXTRA_BYTES disabled as a workaround..."
            )
            from laspy.point.record import (
                PackedPointRecord,  # pylint: disable=import-error
            )

            @classmethod
            def from_buffer_without_extra_bytes(cls, buffer, point_format, count=-1, offset=0):
                item_size = point_format.size
                count = len(buffer) // item_size
                points_dtype = point_format.dtype()
                data = np.frombuffer(buffer, dtype=points_dtype, offset=offset, count=count)
                return cls(data, point_format)

            PackedPointRecord.from_buffer = from_buffer_without_extra_bytes
            las = laspy.read(input_path)
        else:
            raise

    # Use scaled coordinates (scale and offset applied)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    # Build Nx3 point array
    pts = np.vstack((x, y, z)).T

    # Recenter point cloud to reduce floating point precision issues
    shift = pts.mean(axis=0)
    pts -= shift

    # Base PCD fields
    data = {
        "x": pts[:, 0].astype(np.float32),
        "y": pts[:, 1].astype(np.float32),
        "z": pts[:, 2].astype(np.float32),
        "intensity": las.intensity.astype(np.float32),
    }

    # Handle RGB attributes if present
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        # Convert 16-bit LAS colors to 8-bit
        r = (las.red >> 8).astype(np.uint32)
        g = (las.green >> 8).astype(np.uint32)
        b = (las.blue >> 8).astype(np.uint32)

        # Pack RGB into a single float field (PCL-compatible)
        rgb = (r << 16) | (g << 8) | b
        data["rgb"] = rgb.view(np.float32)

    # Write PCD file
    # Create structured array for pypcd4
    field_names = ["x", "y", "z", "intensity"]
    types = [np.float32, np.float32, np.float32, np.float32]

    if "rgb" in data:
        field_names.append("rgb")
        types.append(np.float32)

    arrays = [data[field] for field in field_names]

    pd = pypcd4_pcd.from_points(arrays, field_names, types)
    pd.save(output_path, encoding=Encoding.BINARY_COMPRESSED)
