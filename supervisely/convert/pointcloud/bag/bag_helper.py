def pc2_to_pcd(points, path):
    import open3d as o3d # pylint: disable=import-error

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)
