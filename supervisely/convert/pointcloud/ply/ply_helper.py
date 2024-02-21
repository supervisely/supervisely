import open3d as o3d


def ply2pcd(input_path: str, output_path: str) -> None:
    """Convert .ply format to .pcd."""
    points = o3d.io.read_point_cloud(input_path)
    o3d.io.write_point_cloud(output_path, points, write_ascii=True)