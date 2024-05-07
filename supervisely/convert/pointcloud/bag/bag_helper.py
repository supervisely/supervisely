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


def get_cuboid_from_points(coords: list):
    """Greates a cuboid from a list xyz center, size, and rotation."""
    center, size, rotation = coords

    center = Vector3d(center[0], center[1], center[2])
    size = Vector3d(size[0], size[1], size[2])
    rotation = Vector3d(rotation[0], rotation[1], rotation[2])

    return Cuboid3d(center, rotation, size)


def pc2_to_ann(points: np.ndarray, path: str, meta: ProjectMeta) -> ProjectMeta:
    """Convert a point cloud to an Supervisely annotation file."""
    import open3d as o3d  # pylint: disable=import-error

    figures = []
    objects = PointcloudObjectCollection()
    points = o3d.utility.Vector3dVector(points)

    if len(points) % 3 == 0:
        for i in range(0, len(points), 3):
            coords = np.array([points[j] for j in range(i, i + 3)])

            cuboid = get_cuboid_from_points(coords)
            obj_cls = meta.get_obj_class("object")
            pcd_obj = PointcloudObject(obj_cls)
            objects = objects.add(pcd_obj)
            figure = PointcloudFigure(pcd_obj, cuboid)
            figures.append(figure)

    ann = PointcloudAnnotation(objects=objects, figures=figures)
    dump_json_file(ann.to_json(), path)


def process_vector3_msg(time_to_data, vectors_dict, bag_path, meta, topic, progress_cb):
    """Convert a list of Vector3d to an annotation file."""
    for time, vectors_list in vectors_dict.items():
        objects = PointcloudObjectCollection()
        figures = []
        for i in range(0, len(vectors_list), 3):
            msg_1, msg_2, msg_3 = vectors_list[i : i + 3]
            center = Vector3d(msg_1.vector.x, msg_1.vector.y, msg_1.vector.z)
            size = Vector3d(msg_2.vector.x, msg_2.vector.y, msg_2.vector.z)
            rotation = Vector3d(msg_3.vector.x, msg_3.vector.y, msg_3.vector.z)
            cuboid = Cuboid3d(center, rotation, size)
            obj_cls = meta.get_obj_class("object")
            pcd_obj = PointcloudObject(obj_cls)
            objects = objects.add(pcd_obj)
            figure = PointcloudFigure(pcd_obj, cuboid)
            figures.append(figure)

        ann = PointcloudAnnotation(objects=objects, figures=figures)
        path = bag_path.parent / bag_path.stem / topic / time
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        dump_json_file(ann.to_json(), path.as_posix())
        time_to_data[time]["ann"] = path
        progress_cb(len(vectors_list))


def process_pc2_msg(time_to_data, msg, rostime, bag_path, topic, meta, is_ann=False):
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
