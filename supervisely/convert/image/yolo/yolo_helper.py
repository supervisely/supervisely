import os
import shutil
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import yaml
from tqdm import tqdm

from supervisely._utils import generate_free_name
from supervisely.annotation.annotation import Annotation
from supervisely.annotation.label import Label
from supervisely.geometry.alpha_mask import AlphaMask
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.bitmap import Bitmap
from supervisely.geometry.graph import GraphNodes, KeypointsTemplate, Node
from supervisely.geometry.polygon import Polygon
from supervisely.geometry.polyline import Polyline
from supervisely.geometry.rectangle import Rectangle
from supervisely.imaging.color import generate_rgb
from supervisely.io.fs import get_file_name, get_file_name_with_ext, touch
from supervisely.nn.task_type import TaskType
from supervisely.project.project import Dataset, OpenMode, Project
from supervisely.project.project_meta import ProjectMeta
from supervisely.sly_logger import logger
from supervisely.task.progress import tqdm_sly

YOLO_DETECTION_COORDS_NUM = 4
YOLO_SEGM_MIN_COORDS_NUM = 6
YOLO_KEYPOINTS_MIN_COORDS_NUM = 6


class YOLOTaskType:
    DETECT = "detect"
    SEGMENT = "segment"
    POSE = "pose"


SLY_YOLO_TASK_TYPE_MAP = {
    TaskType.OBJECT_DETECTION: YOLOTaskType.DETECT,
    TaskType.INSTANCE_SEGMENTATION: YOLOTaskType.SEGMENT,
    TaskType.POSE_ESTIMATION: YOLOTaskType.POSE,
}


coco_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def generate_colors(count: int) -> List[List[int]]:
    colors = []
    for _ in range(count):
        new_color = generate_rgb(colors)
        colors.append(new_color)
    return colors


def get_coordinates(line: str) -> Tuple[int, List[float]]:
    """
    Parse coordinates from a line in the YOLO format.
    """
    class_index = int(line[0])
    coords = list(map(float, line[1:]))
    return class_index, coords


def convert_rectangle(
    img_height: int,
    img_width: int,
    coords: List[float],
    **kwargs,
) -> Rectangle:
    """
    Convert rectangle coordinates from relative (0-1) to absolute (px) values.
    """
    x_center, y_center, ann_width, ann_height = coords
    x_center = float(x_center)
    y_center = float(y_center)
    ann_width = float(ann_width)
    ann_height = float(ann_height)

    px_x_center = x_center * img_width
    px_y_center = y_center * img_height

    px_ann_width = ann_width * img_width
    px_ann_height = ann_height * img_height

    left = int(px_x_center - (px_ann_width / 2))
    right = int(px_x_center + (px_ann_width / 2))

    top = int(px_y_center - (px_ann_height / 2))
    bottom = int(px_y_center + (px_ann_height / 2))

    # check if the coordinates are within the image
    left, top = max(0, left), max(0, top)
    right, bottom = min(img_width, right), min(img_height, bottom)

    return Rectangle(top, left, bottom, right)


def validate_polygon_coords(coords: List[float]) -> List[float]:
    """
    Check and correct polygon coordinates:
    - remove the last point if it is the same as the first one
    """
    if coords[0] == coords[-2] and coords[1] == coords[-1]:
        return coords[:-2]
    return coords


def convert_polygon(
    img_height: int,
    img_width: int,
    coords: List[float],
    **kwargs,
) -> Union[Polygon, None]:
    """
    Convert polygon coordinates from relative (0-1) to absolute (px) values.
    """
    coords = validate_polygon_coords(coords)
    if len(coords) < 6:
        logger.warning("Polygon has less than 3 points. Skipping.")
        return None

    exterior = []
    for i in range(0, len(coords), 2):
        x = coords[i]
        y = coords[i + 1]
        px_x = min(img_width, max(0, int(x * img_width)))
        px_y = min(img_height, max(0, int(y * img_height)))
        exterior.append([px_y, px_x])
    return Polygon(exterior=exterior)


def convert_keypoints(
    img_height: int,
    img_width: int,
    num_keypoints: int,
    num_dims: int,
    coords: List[float],
    **kwargs,
) -> Union[GraphNodes, None]:
    """
    Convert keypoints coordinates from relative (0-1) to absolute (px) values.
    """
    nodes = []
    step = 3 if num_dims == 3 else 2
    shift = 4
    for i in range(shift, num_keypoints * step + shift, step):
        x = coords[i]
        y = coords[i + 1]
        visibility = int(coords[i + 2]) if num_dims == 3 else 2
        if visibility in [0, 1]:
            continue  # skip invisible keypoints
        px_x = min(img_width, max(0, int(x * img_width)))
        px_y = min(img_height, max(0, int(y * img_height)))
        node = Node(row=px_y, col=px_x)  # , disabled=v)
        nodes.append(node)
    if len(nodes) > 0:
        return GraphNodes(nodes)


def create_geometry_config(num_keypoints: int = None) -> KeypointsTemplate:
    """
    Create a template for keypoints with the specified number of keypoints.
    """
    i, j = 0, 0
    template = KeypointsTemplate()
    for p in list(range(num_keypoints)):
        template.add_point(label=str(p), row=i, col=j)
        j += 1
        i += 1

    return template


def is_applicable_for_rectangles(coords: List[float], **kwargs) -> bool:
    """
    Check if the coordinates are applicable for rectangles.
    """
    return len(coords) == YOLO_DETECTION_COORDS_NUM


def is_applicable_for_polygons(
    with_keypoint: bool,
    coords: List[float],
    **kwargs,
) -> bool:
    """
    Check if the coordinates are applicable for polygons.

    :param with_keypoint: Whether the YAML config file contains keypoints.
    :type with_keypoint: bool
    """
    if with_keypoint:
        return False
    return len(coords) >= YOLO_SEGM_MIN_COORDS_NUM and len(coords) % 2 == 0


def is_applicable_for_keypoints(
    with_keypoint: bool,
    num_keypoints: int,
    num_dims: int,
    coords: List[float],
    **kwargs,
) -> bool:
    """
    Check if the coordinates are applicable for keypoints.
    """
    if not with_keypoint or not num_keypoints or not num_dims:
        return False
    if len(coords) < YOLO_KEYPOINTS_MIN_COORDS_NUM:
        return False
    return len(coords) == num_keypoints * num_dims + 4


APPLICABLE_GEOMETRIES_MAP = {
    Rectangle: is_applicable_for_rectangles,
    Polygon: is_applicable_for_polygons,
    GraphNodes: is_applicable_for_keypoints,
}


def detect_geometry(
    coords: List[float],
    with_keypoint: bool,
    num_keypoints: int,
    num_dims: int,
) -> Union[Rectangle, Polygon, GraphNodes, None]:
    """
    Detect the geometry type based on the coordinates and the configuration.
    """
    for geometry, is_applicable in APPLICABLE_GEOMETRIES_MAP.items():
        if is_applicable(
            with_keypoint=with_keypoint,
            num_keypoints=num_keypoints,
            num_dims=num_dims,
            coords=coords,
        ):
            return geometry


GEOMETRY_CONVERTERS = {
    Rectangle: convert_rectangle,
    Polygon: convert_polygon,
    GraphNodes: convert_keypoints,
}


def get_geometry(
    geometry_type: Union[Rectangle, Polygon, GraphNodes, AnyGeometry],
    img_height: int,
    img_width: int,
    with_keypoint: bool,
    num_keypoints: int,
    num_dims: int,
    coords: List[float],
) -> Union[Rectangle, Polygon, GraphNodes, None]:
    """
    Get the geometry object based on the geometry type.
    """
    if geometry_type not in GEOMETRY_CONVERTERS:
        geometry_type = detect_geometry(
            coords=coords,
            with_keypoint=with_keypoint,
            num_keypoints=num_keypoints,
            num_dims=num_dims,
        )

    if geometry_type is None:
        return None

    return GEOMETRY_CONVERTERS[geometry_type](
        img_height=img_height,
        img_width=img_width,
        coords=coords,
        num_keypoints=num_keypoints,
        num_dims=num_dims,
    )


def rectangle_to_yolo_line(
    class_idx: int,
    geometry: Rectangle,
    img_height: int,
    img_width: int,
):
    x = geometry.center.col / img_width
    y = geometry.center.row / img_height
    w = geometry.width / img_width
    h = geometry.height / img_height
    return f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def polygon_to_yolo_line(
    class_idx: int,
    geometry: Polygon,
    img_height: int,
    img_width: int,
) -> str:
    coords = []
    for point in geometry.exterior:
        x = point.col / img_width
        y = point.row / img_height
        coords.extend([x, y])
    return f"{class_idx} {' '.join(map(lambda coord: f'{coord:.6f}', coords))}"


def keypoints_to_yolo_line(
    class_idx: int,
    geometry: GraphNodes,
    img_height: int,
    img_width: int,
    max_kpts_count: int,
):
    bbox = geometry.to_bbox()
    x, y, w, h = bbox.center.col, bbox.center.row, bbox.width, bbox.height
    x, y, w, h = x / img_width, y / img_height, w / img_width, h / img_height

    line = f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

    for node in geometry.nodes.values():
        node: Node
        visible = 2 if not node.disabled else 1
        line += (
            f" {node.location.col / img_width:.6f} {node.location.row / img_height:.6f} {visible}"
        )
    if len(geometry.nodes) < max_kpts_count:
        for _ in range(max_kpts_count - len(geometry.nodes)):
            line += " 0 0 0"

    return line


def convert_label_geometry_if_needed(
    label: Label,
    task_type: Literal["detect", "segment", "pose"],
    verbose: bool = False,
) -> List[Label]:
    if task_type == YOLOTaskType.DETECT:
        available_geometry_type = Rectangle
        convertable_geometry_types = [Polygon, GraphNodes, Bitmap, Polyline, AlphaMask, AnyGeometry]
    elif task_type == YOLOTaskType.SEGMENT:
        available_geometry_type = Polygon
        convertable_geometry_types = [Bitmap, AlphaMask, AnyGeometry]
    elif task_type == YOLOTaskType.POSE:
        available_geometry_type = GraphNodes
        convertable_geometry_types = []
    else:
        raise ValueError(
            f"Unsupported task type: {task_type}. "
            f"Supported types: '{YOLOTaskType.DETECT}', '{YOLOTaskType.SEGMENT}', '{YOLOTaskType.POSE}'"
        )

    if label.obj_class.geometry_type == available_geometry_type:
        return [label]

    need_convert = label.obj_class.geometry_type in convertable_geometry_types

    if need_convert:
        new_obj_cls = label.obj_class.clone(geometry_type=available_geometry_type)
        return label.convert(new_obj_cls)

    if verbose:
        logger.warning(
            f"Label '{label.obj_class.name}' has unsupported geometry type: "
            f"{type(label.obj_class.geometry_type)}. Skipping."
        )
    return []


def label_to_yolo_lines(
    label: Label,
    img_height: int,
    img_width: int,
    class_names: List[str],
    task_type: Literal["detect", "segment", "pose"],
) -> List[str]:
    """
    Convert the Supervisely Label to a line in the YOLO format.
    """

    labels = convert_label_geometry_if_needed(label, task_type)
    class_idx = class_names.index(label.obj_class.name)

    lines = []
    for label in labels:
        if task_type == YOLOTaskType.DETECT:
            yolo_line = rectangle_to_yolo_line(
                class_idx=class_idx,
                geometry=label.geometry,
                img_height=img_height,
                img_width=img_width,
            )
        elif task_type == YOLOTaskType.SEGMENT:
            yolo_line = polygon_to_yolo_line(
                class_idx=class_idx,
                geometry=label.geometry,
                img_height=img_height,
                img_width=img_width,
            )
        elif task_type == YOLOTaskType.POSE:
            nodes_field = label.obj_class.geometry_type.items_json_field
            max_kpts_count = len(label.obj_class.geometry_config[nodes_field])
            yolo_line = keypoints_to_yolo_line(
                class_idx=class_idx,
                geometry=label.geometry,
                img_height=img_height,
                img_width=img_width,
                max_kpts_count=max_kpts_count,
            )
        else:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                f"Supported types: '{YOLOTaskType.DETECT}', '{YOLOTaskType.SEGMENT}', '{YOLOTaskType.POSE}'"
            )

        if yolo_line is not None:
            lines.append(yolo_line)

    return lines


def sly_ann_to_yolo(
    ann: Annotation,
    class_names: List[str],
    task_type: Literal["detect", "segment", "pose"] = "detect",
) -> List[str]:
    """
    Convert the Supervisely annotation to the YOLO format.
    """
    h, w = ann.img_size
    yolo_lines = []
    for label in ann.labels:
        lines = label_to_yolo_lines(
            label=label,
            img_height=h,
            img_width=w,
            class_names=class_names,
            task_type=task_type,
        )
        yolo_lines.extend(lines)
    return yolo_lines


def sly_ds_to_yolo(
    dataset: Dataset,
    meta: ProjectMeta,
    dest_dir: Optional[str] = None,
    task_type: Literal["detect", "segment", "pose"] = "detect",
    log_progress: bool = False,
    progress_cb: Optional[Union[tqdm, Callable]] = None,
    is_val: Optional[bool] = None,
) -> str:
    task_type = validate_task_type(task_type)
    if progress_cb is not None:
        log_progress = False

    if log_progress:
        progress_cb = tqdm_sly(
            desc=f"Converting dataset '{dataset.short_name}' to YOLO format",
            total=len(dataset),
        ).update

    dest_dir = Path(dataset.path) / "yolo" if dest_dir is None else Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # * create train and val directories
    images_dir = dest_dir / "images"
    labels_dir = dest_dir / "labels"
    train_images_dir = images_dir / "train"
    train_labels_dir = labels_dir / "train"
    val_images_dir = images_dir / "val"
    val_labels_dir = labels_dir / "val"
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # * convert annotations and copy images
    class_names = [obj_class.name for obj_class in meta.obj_classes]
    used_names = set(os.listdir(train_images_dir)) | set(os.listdir(val_images_dir))
    for name in dataset.get_items_names():
        ann_path = dataset.get_ann_path(name)
        ann = Annotation.load_json_file(ann_path, meta)

        if is_val is not None:
            images_dir = val_images_dir if is_val else train_images_dir
            labels_dir = val_labels_dir if is_val else train_labels_dir
        else:
            images_dir = val_images_dir if ann.img_tags.get("val") else train_images_dir
            labels_dir = val_labels_dir if ann.img_tags.get("val") else train_labels_dir

        img_path = Path(dataset.get_img_path(name))
        img_name = f"{dataset.short_name}_{get_file_name_with_ext(img_path)}"
        img_name = generate_free_name(used_names, img_name, with_ext=True, extend_used_names=True)
        shutil.copy2(img_path, images_dir / img_name)

        label_path = str(labels_dir / f"{get_file_name(img_name)}.txt")
        yolo_lines = ann.to_yolo(class_names, task_type)
        if len(yolo_lines) > 0:
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
        else:
            touch(label_path)

        if progress_cb is not None:
            progress_cb(1)

    # * save data config file if it does not exist
    config_path = dest_dir / "data_config.yaml"
    if not config_path.exists():
        with_keypoint = task_type is YOLOTaskType.POSE
        save_yolo_config(meta, dest_dir, with_keypoint=with_keypoint)

    return str(dest_dir)


def save_yolo_config(meta: ProjectMeta, dest_dir: str, with_keypoint: bool = False):
    dest_dir = Path(dest_dir)
    save_path = dest_dir / "data_config.yaml"
    class_names = [c.name for c in meta.obj_classes]
    class_colors = [c.color for c in meta.obj_classes]
    data_yaml = {
        "train": f"../{str(dest_dir.name)}/images/train",
        "val": f"../{str(dest_dir.name)}/images/val",
        "train_labels": f"../{str(dest_dir.name)}/labels/train",
        "val_labels": f"../{str(dest_dir.name)}/labels/val",
        "nc": len(class_names),
        "names": class_names,
        "colors": class_colors,
    }
    has_keypoints = any(c.geometry_type == GraphNodes for c in meta.obj_classes)
    if has_keypoints and with_keypoint:
        max_kpts_count = 0
        for obj_class in meta.obj_classes:
            if issubclass(obj_class.geometry_type, GraphNodes):
                field_name = obj_class.geometry_type.items_json_field
                max_kpts_count = max(max_kpts_count, len(obj_class.geometry_config[field_name]))
        data_yaml["kpt_shape"] = [max_kpts_count, 3]
        data_yaml["flip_idx"] = [i for i in range(max_kpts_count)]
    with open(save_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=None)

    logger.info(f"Data config file has been saved to {str(save_path)}")


def sly_project_to_yolo(
    project: Union[Project, str],
    dest_dir: Optional[str] = None,
    task_type: Literal["detect", "segment", "pose"] = "detect",
    log_progress: bool = False,
    progress_cb: Optional[Callable] = None,
    val_datasets: Optional[List[str]] = None,
):
    """
    Convert Supervisely project to YOLO format.

    :param project: Supervisely project or path to the directory with the project.
    :type project: :class:`supervisely.project.project.Project` or :class:`str`
    :param dest_dir: Destination directory.
    :type dest_dir: :class:`str`, optional
    :param task_type: Task type.
    :type task_type: :class:`str`, optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: :class:`bool`
    :param progress_cb: Function for tracking conversion progress (for all items in the project).
    :type progress_cb: callable, optional
    :param val_datasets:    List of dataset names for validation.
                            Full dataset names are required (e.g., 'ds0/nested_ds1/ds3').
                            If specified, datasets from the list will be marked as val, others as train.
                            If not specified, the function will determine the validation datasets automatically.
    :type val_datasets: :class:`list`, optional
    :return: Path to the destination directory.
    :rtype: :class:`str`

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        # Local folder with Project
        project_directory = "/home/admin/work/supervisely/source/project"

        # Convert Project to YOLO format
        sly.Project(project_directory).to_yolo(log_progress=True)
    """
    task_type = validate_task_type(task_type)
    if isinstance(project, str):
        project = Project(project, mode=OpenMode.READ)

    dest_dir = Path(project.directory).parent / "yolo" if dest_dir is None else Path(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    if len(os.listdir(dest_dir)) > 0:
        raise FileExistsError(f"Directory {dest_dir} is not empty.")

    if progress_cb is not None:
        log_progress = False

    if log_progress:
        progress_cb = tqdm_sly(
            desc="Converting Supervisely project to YOLO format", total=project.total_items
        ).update

    with_keypoint = task_type is YOLOTaskType.POSE
    save_yolo_config(project.meta, dest_dir, with_keypoint=with_keypoint)

    for dataset in project.datasets:
        if val_datasets is not None:
            is_val = dataset.name in val_datasets
        else:
            is_val = None

        dataset: Dataset
        dataset.to_yolo(
            meta=project.meta,
            dest_dir=dest_dir,
            task_type=task_type,
            log_progress=log_progress,
            progress_cb=progress_cb,
            is_val=is_val,
        )
        logger.info(f"Dataset '{dataset.short_name}' has been converted to YOLO format.")
    logger.info(f"Project '{project.name}' has been converted to YOLO format.")

    return str(dest_dir)


def to_yolo(
    input_data: Union[Project, Dataset, str],
    dest_dir: Optional[str] = None,
    task_type: Literal["detect", "segment", "pose"] = "detect",
    meta: Optional[ProjectMeta] = None,
    log_progress: bool = True,
    progress_cb: Optional[Callable] = None,
    val_datasets: Optional[List[str]] = None,
    is_val: Optional[bool] = None,
) -> Union[None, str]:
    """
    Universal function to convert Supervisely project or dataset  to YOLO format.
    Note:
        - For better compatibility, please pass named arguments explicitly. Otherwise, the function may not work as expected.
            You can use the dedicated functions for each data type:

                - :func:`sly.convert.sly_project_to_yolo()`
                - :func:`sly.convert.sly_ds_to_yolo()`

        - If the input_data is a Project, the dest_dir parameters are required.
        - If the input_data is a Dataset, the meta and dest_dir parameters are required.

    :param input_data: Supervisely project or dataset, or path to the directory with the project/dataset.
    :type input_data: :class:`supervisely.project.project.Project`, :class:`supervisely.project.dataset.Dataset`, or :class:`str`
    :param dest_dir: Destination directory.
    :type dest_dir: :class:`str`, optional
    :param task_type: Task type.
    :type task_type: :class:`str`, optional
    :param meta: Project meta (required for Dataset conversion).
    :type meta: :class:`supervisely.project.project_meta.ProjectMeta`, optional
    :param log_progress: Show uploading progress bar.
    :type log_progress: :class:`bool`
    :param progress_cb: Function for tracking conversion progress (for all items in the project).
    :type progress_cb: callable, optional
    :param val_datasets:    List of dataset names for validation.
                            Full dataset names are required (e.g., 'ds0/nested_ds1/ds3').
                            If specified, datasets from the list will be marked as val, others as train.
                            If not specified, the function will determine the validation datasets automatically.
    :type val_datasets: :class:`list`, optional
    :param is_val: Whether the dataset is for validation.
    :type is_val: :class:`bool`, optional
    :return: None, list of YOLO lines, or path to the destination directory.
    :rtype: NoneType, list, str

    :Usage example:

    .. code-block:: python

        import supervisely as sly

        # Local folder with Project
        project_directory = "/home/admin/work/supervisely/source/project"
        project_fs = sly.Project(project_directory, sly.OpenMode.READ)

        # Convert Project to YOLO format
        sly.convert.to_yolo(project_directory, dest_dir="./yolo")
        # or
        sly.convert.to_yolo(project_fs, dest_dir="./yolo")

        # Convert Dataset to YOLO format
        dataset: sly.Dataset = project_fs.datasets.get("dataset_name")
        sly.convert.to_yolo(dataset, dest_dir="./yolo", meta=project_fs.meta, is_val=True)
    """
    if isinstance(input_data, str):
        try:
            input_data = Project(input_data, mode=OpenMode.READ)
        except Exception:
            try:
                input_data = Dataset(input_data, mode=OpenMode.READ)
            except Exception:
                raise ValueError("Please check the path or the input data.")
    if isinstance(input_data, Project):
        return sly_project_to_yolo(
            project=input_data,
            dest_dir=dest_dir,
            task_type=task_type,
            log_progress=log_progress,
            progress_cb=progress_cb,
            val_datasets=val_datasets,
        )
    elif isinstance(input_data, Dataset):
        return sly_ds_to_yolo(
            dataset=input_data,
            meta=meta,
            dest_dir=dest_dir,
            task_type=task_type,
            log_progress=log_progress,
            progress_cb=progress_cb,
            is_val=is_val,
        )
    else:
        raise ValueError("Unsupported input type. Only Project or Dataset are supported.")


def validate_task_type(task_type: Literal["detect", "segment", "pose"]) -> str:
    if task_type not in [YOLOTaskType.DETECT, YOLOTaskType.SEGMENT, YOLOTaskType.POSE]:
        task_type = SLY_YOLO_TASK_TYPE_MAP.get(task_type)
        if task_type is None:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                f"Supported types: '{YOLOTaskType.DETECT}', '{SLY_YOLO_TASK_TYPE_MAP[TaskType.OBJECT_DETECTION]}', "
                f"'{YOLOTaskType.SEGMENT}', '{SLY_YOLO_TASK_TYPE_MAP[TaskType.INSTANCE_SEGMENTATION]}', "
                f"'{YOLOTaskType.POSE}', '{SLY_YOLO_TASK_TYPE_MAP[TaskType.POSE_ESTIMATION]}'"
            )
    return task_type
