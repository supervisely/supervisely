from supervisely import PointcloudAnnotation, dump_json_file


def process_pointcloud_msg(time_to_data, pointcloud, timestamp, file_path, meta, is_ann=False):
    """Process a pointcloud message and save it as a PCD file or JSON annotation file."""
    points = pointcloud[:, :3]  # Extract x, y, z coordinates
    intensities = pointcloud[:, 3]  # Extract intensity values

    time = f"{timestamp:.6f}"
    name = f"{time}.pcd" if not is_ann else f"{time}.pcd.json"
    path = file_path.parent / file_path.stem / name
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if is_ann:
        ann = PointcloudAnnotation()
        dump_json_file(ann.to_json(), path.as_posix())
        time_to_data[time]["ann"] = path
    else:
        pcd_meta = {"frame_id": "lidar", "time": time}
        time_to_data[time]["meta"] = pcd_meta
        time_to_data[time]["ann"] = None

        with open(path, "w") as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z intensity\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {points.shape[0]}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {points.shape[0]}\n")
            f.write("DATA ascii\n")
            for i in range(points.shape[0]):
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {intensities[i]}\n")
        time_to_data[time]["pcd"] = path
