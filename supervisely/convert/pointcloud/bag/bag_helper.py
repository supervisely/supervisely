import math

import numpy as np

from supervisely import ProjectMeta
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.point_3d import Vector3d
from supervisely.io.json import dump_json_file
from supervisely.pointcloud_annotation.pointcloud_annotation import PointcloudAnnotation
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)


def pc2_to_pcd(points, path):
    """Convert a point cloud to a PCD file."""

    import open3d as o3d  # pylint: disable=import-error

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)


def get_cuboid_from_points(points: list):
    """Get a cuboid from a list of 8 points."""

    all_x = list(sorted(set([point[0] for point in points])))
    all_y = list(sorted(set([point[1] for point in points])))
    all_z = list(sorted(set([point[2] for point in points])))
    min_x, min_y, min_z = all_x[0], all_y[0], all_z[0]
    max_x, max_y, max_z = all_x[-1], all_y[-1], all_z[-1]

    center = Vector3d((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
    size = Vector3d(max_x - min_x, max_y - min_y, max_z - min_z)
    yaw = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    pitch = math.atan2(points[2][2] - points[0][2], points[2][0] - points[0][0])
    roll = math.atan2(points[4][2] - points[0][2], points[4][1] - points[0][1])

    rotation = Vector3d(pitch, roll, yaw)
    cuboid = Cuboid3d(center, rotation, size)
    return cuboid


def pc2_to_ann(points: np.ndarray, path: str, meta: ProjectMeta) -> ProjectMeta:
    """Convert a point cloud to an Supervisely annotation file."""
    import open3d as o3d  # pylint: disable=import-error

    figures = []
    objects = PointcloudObjectCollection()
    points = o3d.utility.Vector3dVector(points)

    if len(points) % 8 == 0:
        for i in range(0, len(points), 8):
            corners = [points[j] for j in range(i, i + 8)]

            cuboid = get_cuboid_from_points(corners)
            obj_cls = meta.get_obj_class("object")
            pcd_obj = PointcloudObject(obj_cls)
            objects = objects.add(pcd_obj)
            figure = PointcloudFigure(pcd_obj, cuboid)
            figures.append(figure)

    ann = PointcloudAnnotation(objects=objects, figures=figures)
    dump_json_file(ann.to_json(), path)


def process_msg(time_to_data, msg, rostime, bag_path, topic, meta, is_ann=False):
    """Process a ROS message and save it as a PCD file or JSON annotation file."""
    import sensor_msgs.point_cloud2 as pc2  # pylint: disable=import-error

    p_ = []
    gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))

    for p in gen:
        p_.append(p)

    time = f"{rostime.secs}.{rostime.nsecs}"
    name = f"{time}.pcd" if not is_ann else f"{time}.pcd.json"
    path = bag_path.parent / bag_path.stem / topic / name
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if is_ann:
        pc2_to_ann(p_, path.as_posix(), meta)
        time_to_data[time]["ann"] = path
    else:
        pcd_meta = {"frame_id": msg.header.frame_id, "time": time}
        time_to_data[time]["meta"] = pcd_meta
        time_to_data[time]["ann"] = None

        pc2_to_pcd(p_, path.as_posix())
        time_to_data[time]["pcd"] = path
