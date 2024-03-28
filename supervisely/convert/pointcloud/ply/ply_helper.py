def ply2pcd(input_path: str, output_path: str) -> None:
    """Convert .ply format to .pcd."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "No module named open3d. Please make sure that module is installed from pip and try again."
        )

    points = o3d.io.read_point_cloud(input_path)
    o3d.io.write_point_cloud(output_path, points, write_ascii=True)