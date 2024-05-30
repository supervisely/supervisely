from supervisely import logger

import numpy as np


def las2pcd(input_path, output_path):
    import laspy # pylint: disable=import-error
    import open3d as o3d # pylint: disable=import-error

    try:
        las = laspy.read(input_path)
    except Exception as e:
        if "buffer size must be a multiple of element size" in str(e):
            from laspy.point.record import PackedPointRecord # pylint: disable=import-error
            logger.warn(
                "Could not read LAS file in laspy. Trying to read it without EXTRA_BYTES..."
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
            raise e
    point_cloud = np.vstack((las.X, las.Y, las.Z)).T
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    o3d.io.write_point_cloud(output_path, pc)