from supervisely import logger
from supervisely.io.fs import get_file_name
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.geometry.point_3d import Vector3d
from supervisely.geometry.point import Point

from collections import defaultdict
import os
import numpy as np


MAX_N = 1000


def local2global(semanticId, instanceId):
    globalId = semanticId * MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int)
    else:
        return int(globalId)


def global2local(globalId):
    semanticId = globalId // MAX_N
    instanceId = globalId % MAX_N
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(int), instanceId.astype(int)
    else:
        return int(semanticId), int(instanceId)


annotation2global = defaultdict()


# Abstract base class for annotation objects
class KITTI360Object:
    from abc import ABCMeta

    __metaclass__ = ABCMeta

    def __init__(self):
        from matplotlib import cm

        # the label
        self.label = ""

        # colormap
        self.cmap = cm.get_cmap("Set1")
        self.cmap_length = 9

    def getColor(self, idx):
        if idx == 0:
            return np.array([0, 0, 0])
        return np.asarray(self.cmap(idx % self.cmap_length)[:3]) * 255.0

    # def assignColor(self):
    #     from kitti360scripts.helpers.labels import id2label  # pylint: disable=import-error

    #     if self.semanticId >= 0:
    #         self.semanticColor = id2label[self.semanticId].color
    #         if self.instanceId > 0:
    #             self.instanceColor = self.getColor(self.instanceId)
    #         else:
    #             self.instanceColor = self.semanticColor


# Class that contains the information of a single annotated object as 3D bounding box
class KITTI360Bbox3D(KITTI360Object):
    # Constructor
    def __init__(self):
        KITTI360Object.__init__(self)
        # the polygon as list of points
        self.vertices = []
        self.faces = []
        self.lines = [
            [0, 5],
            [1, 4],
            [2, 7],
            [3, 6],
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
        ]

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # the window that contains the bbox
        self.start_frame = -1
        self.end_frame = -1

        # timestamp of the bbox (-1 if statis)
        self.timestamp = -1

        # projected vertices
        self.vertices_proj = None
        self.meshes = []

        # name
        self.name = ""

    def __str__(self):
        return self.name

    # def generateMeshes(self):
    #     self.meshes = []
    #     if self.vertices_proj:
    #         for fidx in range(self.faces.shape[0]):
    #             self.meshes.append(
    #                 [
    #                     Point(self.vertices_proj[0][int(x)], self.vertices_proj[1][int(x)])
    #                     for x in self.faces[fidx]
    #                 ]
    #             )

    def parseOpencvMatrix(self, node):
        rows = int(node.find("rows").text)
        cols = int(node.find("cols").text)
        data = node.find("data").text.split(" ")

        mat = []
        for d in data:
            d = d.replace("\n", "")
            if len(d) < 1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find("transform"))
        R = transform[:3, :3]
        T = transform[:3, 3]
        vertices = self.parseOpencvMatrix(child.find("vertices"))
        faces = self.parseOpencvMatrix(child.find("faces"))

        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        self.vertices = vertices
        self.faces = faces
        self.R = R
        self.T = T

        self.transform = transform

    def parseBbox(self, child):
        from kitti360scripts.helpers.labels import kittiId2label  # pylint: disable=import-error

        semanticIdKITTI = int(child.find("semanticId").text)
        self.semanticId = kittiId2label[semanticIdKITTI].id
        self.instanceId = int(child.find("instanceId").text)
        # self.name = str(child.find('label').text)
        self.name = kittiId2label[semanticIdKITTI].name

        self.start_frame = int(child.find("start_frame").text)
        self.end_frame = int(child.find("end_frame").text)

        self.timestamp = int(child.find("timestamp").text)

        self.annotationId = int(child.find("index").text) + 1

        global annotation2global
        annotation2global[self.annotationId] = local2global(self.semanticId, self.instanceId)
        self.parseVertices(child)

    def parseStuff(self, child):
        from kitti360scripts.helpers.labels import name2label  # pylint: disable=import-error

        classmap = {
            "driveway": "parking",
            "ground": "terrain",
            "unknownGround": "ground",
            "railtrack": "rail track",
        }
        label = child.find("label").text
        if label in classmap.keys():
            label = classmap[label]

        self.start_frame = int(child.find("start_frame").text)
        self.end_frame = int(child.find("end_frame").text)

        self.semanticId = name2label[label].id
        self.instanceId = 0
        self.parseVertices(child)


# Class that contains the information of the point cloud a single frame
class KITTI360Point3D(KITTI360Object):
    # Constructor
    def __init__(self):
        KITTI360Object.__init__(self)

        self.vertices = []

        self.vertices_proj = None

        # the ID of the corresponding object
        self.semanticId = -1
        self.instanceId = -1
        self.annotationId = -1

        # name
        self.name = ""

        # color
        self.semanticColor = None
        self.instanceColor = None

    def __str__(self):
        return self.name

    # def generateMeshes(self):
    #     pass


# Meta class for KITTI360Bbox3D
class Annotation3D:
    def __init__(self, labelPath):
        from kitti360scripts.helpers.labels import labels  # pylint: disable=import-error
        import xml.etree.ElementTree as ET

        key_name = get_file_name(labelPath)
        # load annotation
        tree = ET.parse(labelPath)
        root = tree.getroot()

        self.objects = defaultdict(dict)

        self.num_bbox = 0
        for child in root:
            if child.find("transform") is None:
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            globalId = local2global(obj.semanticId, obj.instanceId)
            self.objects[globalId][obj.timestamp] = obj
            self.num_bbox += 1

        globalIds = np.asarray(list(self.objects.keys()))
        semanticIds, instanceIds = global2local(globalIds)
        for label in labels:
            if label.hasInstances:
                print(f"{label.name:<30}:\t {(semanticIds==label.id).sum()}")
        print(f"Loaded {len(globalIds)} instances")
        print(f"Loaded {self.num_bbox} boxes")

    def __call__(self, semanticId, instanceId, timestamp=None):
        globalId = local2global(semanticId, instanceId)
        if globalId in self.objects.keys():
            # static object
            if len(self.objects[globalId].keys()) == 1:
                if -1 in self.objects[globalId].keys():
                    return self.objects[globalId][-1]
                else:
                    return None
            # dynamic object
            else:
                return self.objects[globalId][timestamp]
        else:
            return None

    def get_objects(self):
        return [list(obj.values())[0] for obj in self.objects.values()]

class StaticTransformations:
    def __init__(self, calibrations_path):
        import kitti360scripts.devkits.commons.loadCalibration as lc  # pylint: disable=import-error

        cam2velo_path = os.path.join(calibrations_path, "calib_cam_to_velo.txt")
        self.cam2velo = lc.loadCalibrationRigid(cam2velo_path)
        perspective_path = os.path.join(calibrations_path, "perspective.txt")
        self.intrinsic_calibrations = lc.loadPerspectiveIntrinsic(perspective_path)
        self.cam2world = None

    def set_cam2world(self, cam2world_path):
        if not os.path.isfile(cam2world_path):
            logger.warn("Camera to world calibration file was not found")
            return

        cam2world_rows = np.loadtxt(cam2world_path)
        cam2world_rigid = np.reshape(cam2world_rows[:, 1:], (-1, 4, 4))
        frames_numbers = list(np.reshape(cam2world_rows[:, :1], (-1)).astype(int))
        cam2world = {}

        current_rigid = cam2world_rigid[0]

        for frame_index in range(0, frames_numbers[-1]):
            if frame_index in frames_numbers:
                mapped_index = frames_numbers.index(frame_index)
                current_rigid = cam2world_rigid[mapped_index]

            # (Tr(cam -> world))
            cam2world[frame_index] = current_rigid
        self.cam2world = cam2world

    def world_to_velo_transformation(self, obj, frame_index):
        # rotate_z = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()
        # rotate_z = np.hstack((rotate_z, np.asarray([[0, 0, 0]]).T))

        # tr0(local -> fixed_coordinates_local)
        tr0 = np.asarray([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # tr0(fixed_coordinates_local -> world)
        tr1 = obj.transform

        # tr2(world -> cam)
        tr2 = np.linalg.inv(self.cam2world[frame_index])

        # tr3(world -> cam)
        tr3 = self.cam2velo

        return tr3 @ tr2 @ tr1 @ tr0

    def get_extrinsic_matrix(self):
        return np.linalg.inv(self.cam2velo)[:3, :4]

    def get_intrinsics_matrix(self, camera_num):
        try:
            matrix = self.intrinsic_calibrations[f"P_rect_0{camera_num}"][:3, :3]
            return matrix
        except KeyError:
            logger.warn(f"Camera {camera_num} intrinsic matrix was not found")
        return

def convert_kitti_cuboid_to_supervisely_geometry(tr_matrix):
    import transforms3d  # pylint: disable=import-error
    from scipy.spatial.transform.rotation import Rotation

    Tdash, Rdash, Zdash, _ = transforms3d.affines.decompose44(tr_matrix)

    x, y, z = Tdash[0], Tdash[1], Tdash[2]
    position = Vector3d(x, y, z)

    rotation_angles = Rotation.from_matrix(Rdash).as_euler("xyz", degrees=False)
    r_x, r_y, r_z = rotation_angles[0], rotation_angles[1], rotation_angles[2]

    # Invert the bbox by adding π to the yaw while maintaining its degree relative to the world
    rotation = Vector3d(r_x, r_y, r_z + np.pi)

    w, h, l = Zdash[0], Zdash[1], Zdash[2]
    dimension = Vector3d(w, h, l)

    return Cuboid3d(position, rotation, dimension)

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


def convert_calib_to_image_meta(image_name, static, cam_num):
    intrinsic_matrix = static.get_intrinsics_matrix(cam_num)
    extrinsic_matrix = static.get_extrinsic_matrix()

    data = {
        "name": image_name,
        "meta": {
            "deviceId": cam_num,
            "sensorsData": {
                "extrinsicMatrix": list(extrinsic_matrix.flatten().astype(float)),
                "intrinsicMatrix": list(intrinsic_matrix.flatten().astype(float)),
            },
        },
    }
    return data
