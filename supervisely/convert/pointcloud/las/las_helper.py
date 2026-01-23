import numpy as np

from supervisely import logger
from supervisely.io.fs import get_file_name_with_ext


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
        input_file_name = get_file_name_with_ext(input_path)
        logger.info(f"Start processing file: {input_file_name}")
        las = laspy.read(input_path)
    except Exception as e:
        if "buffer size must be a multiple of element size" in str(e):
            logger.warning(
                f"{input_file_name} file read failed due to buffer size mismatch with EXTRA_BYTES. "
                "Retrying with EXTRA_BYTES disabled as a workaround..."
            )
            from laspy.point.record import PackedPointRecord  # pylint: disable=import-error

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
            logger.error(f"Failed to read {input_file_name}: {e}")
            return

    try:
        # Use scaled coordinates (scale and offset applied)
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)

        # Check for empty point cloud
        if len(x) == 0:
            logger.warning(f"{input_file_name} file is empty (0 points).")
            return

        # Recenter point cloud to reduce floating point precision issues
        # Calculate shift for each axis independently (avoids creating intermediate pts array)
        shift_x = x.mean()
        shift_y = y.mean()
        shift_z = z.mean()

        logger.info(
            f"Applied coordinate shift for {input_file_name}: "
            f"X={shift_x}, Y={shift_y}, Z={shift_z}"
        )

        # Base PCD fields - apply shift and convert to float32 in one operation
        data = {
            "x": (x - shift_x).astype(np.float32),
            "y": (y - shift_y).astype(np.float32),
            "z": (z - shift_z).astype(np.float32),
            "intensity": las.intensity.astype(np.float32),
        }

        # Handle RGB attributes if present
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            # Convert LAS colors to 8-bit.
            # Some files store 0–255 values in 16-bit fields; detect this and only shift when needed.
            r_raw = np.asarray(las.red)
            g_raw = np.asarray(las.green)
            b_raw = np.asarray(las.blue)

            # Determine if the values are full 16-bit range (0–65535) or already 0–255.
            max_rgb = max(
                r_raw.max(initial=0),
                g_raw.max(initial=0),
                b_raw.max(initial=0),
            )

            if max_rgb > 255:
                # Typical LAS case: 16-bit colors; downscale to 8-bit.
                r = (r_raw >> 8).astype(np.uint32)
                g = (g_raw >> 8).astype(np.uint32)
                b = (b_raw >> 8).astype(np.uint32)
            else:
                # Values are already in 0–255 range; use as-is.
                r = r_raw.astype(np.uint32)
                g = g_raw.astype(np.uint32)
                b = b_raw.astype(np.uint32)

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
    except Exception as e:
        logger.error(f"Error processing {input_file_name}: {e}")
        return

    try:
        pd = pypcd4_pcd.from_points(arrays, field_names, types)
        pd.save(output_path, encoding=Encoding.BINARY_COMPRESSED)
    except Exception as e:
        logger.error(f"Failed to write PCD file for {input_file_name}: {e}")
        return
