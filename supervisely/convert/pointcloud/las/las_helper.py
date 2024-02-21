import laspy
import numpy as np
import open3d as o3d


def las2pcd(input_path, output_path):
    las = laspy.read(input_path)
    point_cloud = np.vstack((las.X, las.Y, las.Z)).T
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    o3d.io.write_point_cloud(output_path, pc)