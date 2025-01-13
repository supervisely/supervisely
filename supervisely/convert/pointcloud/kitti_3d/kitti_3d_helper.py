import numpy as np

from supervisely import ObjClass, ObjClassCollection, ProjectMeta
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.point_3d import Vector3d
from supervisely.pointcloud_annotation.pointcloud_figure import PointcloudFigure
from supervisely.pointcloud_annotation.pointcloud_object import PointcloudObject

FOLDER_NAMES = ["velodyne", "image_2", "label_2", "calib"]


def read_kitti_label(label_path, calib_path):
    import open3d as o3d  # pylint: disable=import-error

    calib = o3d.ml.datasets.KITTI.read_calib(calib_path)
    label = o3d.ml.datasets.KITTI.read_label(label_path, calib)
    return label


def convert_labels_to_meta(labels):
    labels = flatten(labels)
    unique_labels = np.unique([l.label_class for l in labels])
    obj_classes = [ObjClass(k, Cuboid3d) for k in unique_labels]
    meta = ProjectMeta(obj_classes=ObjClassCollection(obj_classes))
    return meta


def convert_bin_to_pcd(src, dst):
    import open3d as o3d  # pylint: disable=import-error

    try:
        bin = np.fromfile(src, dtype=np.float32).reshape(-1, 4)
    except ValueError as e:
        raise Exception(
            f"Incorrect data in the KITTI 3D pointcloud file: {src}. "
            f"There was an error while trying to reshape the data into a 4-column matrix: {e}. "
            "Please ensure that the binary file contains a multiple of 4 elements to be "
            "successfully reshaped into a (N, 4) array.\n"
        )
    points = bin[:, 0:3]
    intensity = bin[:, -1]
    intensity_fake_rgb = np.zeros((intensity.shape[0], 3))
    intensity_fake_rgb[:, 0] = intensity
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(intensity_fake_rgb)
    o3d.io.write_point_cloud(dst, pc)


def flatten(list_2d):
    return sum(list_2d, [])


def _convert_label_to_geometry(label):
    geometries = []
    for l in label:
        bbox = l.to_xyzwhlr()
        dim = bbox[[3, 5, 4]]
        pos = bbox[:3] + [0, 0, dim[1] / 2]
        yaw = bbox[-1]
        position = Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
        rotation = Vector3d(0, 0, float(-yaw))

        dimension = Vector3d(float(dim[0]), float(dim[2]), float(dim[1]))
        geometry = Cuboid3d(position, rotation, dimension)
        geometries.append(geometry)
    return geometries


def convert_label_to_annotation(label, meta, renamed_class_names: dict = None):
    geometries = _convert_label_to_geometry(label)
    figures = []
    objs = []
    for l, geometry in zip(label, geometries):  # by object in point cloud
        class_name = renamed_class_names.get(l.label_class, l.label_class)
        pcobj = PointcloudObject(meta.get_obj_class(class_name))
        figures.append(PointcloudFigure(pcobj, geometry))
        objs.append(pcobj)

    return objs, figures


def convert_calib_to_image_meta(image_name, calib_path, camera_num=2):
    with open(calib_path, "r") as f:
        lines = f.readlines()

    assert 0 < camera_num < 4
    intrinsic_matrix = lines[camera_num].strip().split(" ")[1:]
    intrinsic_matrix = np.array(intrinsic_matrix, dtype=np.float32).reshape(3, 4)[:3, :3]

    obj = lines[4].strip().split(" ")[1:]
    rect_4x4 = np.eye(4, dtype=np.float32)
    rect_4x4[:3, :3] = np.array(obj, dtype=np.float32).reshape(3, 3)

    obj = lines[5].strip().split(" ")[1:]
    Tr_velo_to_cam = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam[:3] = np.array(obj, dtype=np.float32).reshape(3, 4)
    world_cam = np.transpose(rect_4x4 @ Tr_velo_to_cam)
    extrinsic_matrix = world_cam[:4, :3].T

    data = {
        "name": image_name,
        "meta": {
            "deviceId": "CAM_LEFT",
            "sensorsData": {
                "extrinsicMatrix": list(extrinsic_matrix.flatten().astype(float)),
                "intrinsicMatrix": list(intrinsic_matrix.flatten().astype(float)),
            },
        },
    }
    return data
