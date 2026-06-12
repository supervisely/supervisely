import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from supervisely import ProjectMeta, logger
from supervisely._utils import generate_free_name, is_development
from supervisely.api.api import Api
from supervisely.convert.base_converter import AvailableMeshConverters
from supervisely.convert.mesh.mesh_converter import MeshConverter
from supervisely.geometry.any_geometry import AnyGeometry
from supervisely.geometry.mesh import Mesh
from supervisely.io.fs import get_file_ext
from supervisely.io.json import load_json_file
from supervisely.project.project_type import ProjectType

# Keys used in _project_structure (mirrors SLYMeshConverter / SLYImageConverter).
_DATASET_ITEMS = "items"
_NESTED_DATASETS = "datasets"

ANN_DIR_NAME = "ann"
GEOMETRIES_DIR_NAME = "geometries"
PLY_EXT = ".ply"
UNLABELED_ID = -1

# Neutral color written over label paint when the mesh keeps its color properties
# (some vertices carry original, non-label colors that must be preserved).
_NEUTRAL_COLOR = "255"


@dataclass
class PLYLabels:
    groups: Dict[Tuple[str, Optional[int]], List[int]]
    has_rgb: bool
    labeled_vertices: int


class PerVertexLabelsMeshConverter(MeshConverter):
    """Imports ASCII PLY meshes whose vertex RGB values encode class labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = False
        # Populated when the input contains more than one dataset directory.
        # Structure: { ds_name: { "items": [Item, ...], "datasets": { child: {...} } } }
        self._project_structure = None

    def __str__(self) -> str:
        return AvailableMeshConverters.PER_VERTEX_LABELS

    @property
    def ann_ext(self) -> str:
        return None

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta = None) -> bool:
        return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        if self.upload_as_links and self.supports_links:
            self._download_remote_ann_files(exts_to_download=[self.key_file_ext])

        try:
            project_dir = self._find_project_dir(self._input_data)
            if project_dir is None or self._has_sly_annotation_sidecars(project_dir):
                return False

            meta_path = os.path.join(project_dir, "meta.json")
            if not self.validate_key_file(meta_path):
                return False
            if self._meta.project_type not in (None, ProjectType.MESHES.value):
                return False

            color_to_class = self._build_color_to_class(self._meta)
            mesh_paths = self._find_mesh_paths(project_dir)
            if len(mesh_paths) == 0:
                return False

            project = {}
            items = []
            labeled_items = 0
            ds_names_seen = set()

            for mesh_path in mesh_paths:
                labels = _read_ascii_ply_labels(mesh_path, color_to_class)
                if not labels.has_rgb:
                    return False
                ann_json = self._labels_to_annotation(labels)
                if labels.labeled_vertices > 0:
                    labeled_items += 1
                item = self.Item(item_path=mesh_path, ann_data=ann_json)
                items.append(item)

                # Determine dataset from the item's directory relative to project_dir.
                # e.g. project_dir/parent/child1/box.ply → "parent/child1"
                rel_dir = os.path.relpath(os.path.dirname(mesh_path), project_dir)
                if rel_dir == ".":
                    rel_dir = "ds0"  # items directly in project_dir → default dataset name
                ds_name_posix = Path(rel_dir).as_posix()
                ds_names_seen.add(ds_name_posix)

                parts = ds_name_posix.split("/")
                curr_ds = project.setdefault(
                    parts[0], {_DATASET_ITEMS: [], _NESTED_DATASETS: {}}
                )
                for part in parts[1:]:
                    curr_ds = curr_ds[_NESTED_DATASETS].setdefault(
                        part, {_DATASET_ITEMS: [], _NESTED_DATASETS: {}}
                    )
                curr_ds[_DATASET_ITEMS].append(item)

            if labeled_items == 0:
                return False

            self._items = items
            has_nested = len(ds_names_seen) > 1 or any("/" in ds for ds in ds_names_seen)
            if has_nested:
                self._project_structure = project
            return True
        except Exception as e:
            logger.info(f"Failed to read Per-Vertex Labels mesh project: {repr(e)}")
            self._items = []
            self._meta = None
            return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 10,
        log_progress=True,
    ) -> None:
        self._strip_label_data_from_meshes()
        if self._project_structure:
            self.upload_project(api, dataset_id, batch_size, log_progress)
        else:
            super().upload_dataset(api, dataset_id, batch_size, log_progress)

    def upload_project(
        self, api: Api, dataset_id: int, batch_size: int = 10, log_progress=True
    ) -> None:
        """Upload a multi-dataset per-vertex-labels project preserving the directory hierarchy."""
        dataset_info = api.dataset.get_info_by_id(dataset_id, raise_error=True)
        project_id = dataset_info.project_id
        existing_datasets = {
            ds.name for ds in api.dataset.get_list(project_id, recursive=True)
        }

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading meshes...")
        else:
            progress, progress_cb = None, None

        def _upload_project(
            project_structure: Dict,
            project_id: int,
            dataset_id: int,
            parent_id: Optional[int] = None,
            first_dataset: bool = False,
        ):
            for ds_name, value in project_structure.items():
                ds_name = generate_free_name(existing_datasets, ds_name, extend_used_names=True)
                if first_dataset:
                    first_dataset = False
                    api.dataset.update(dataset_id, ds_name)
                else:
                    dataset_id = api.dataset.create(project_id, ds_name, parent_id=parent_id).id

                items = value.get(_DATASET_ITEMS, [])
                nested = value.get(_NESTED_DATASETS, {})
                logger.info(
                    f"Dataset: {ds_name}, items: {len(items)}, nested datasets: {len(nested)}"
                )
                if items:
                    super(PerVertexLabelsMeshConverter, self).upload_dataset(
                        api, dataset_id, batch_size, entities=items, progress_cb=progress_cb
                    )
                if nested:
                    _upload_project(nested, project_id, dataset_id, dataset_id)

        _upload_project(self._project_structure, project_id, dataset_id, first_dataset=True)

        if is_development() and progress is not None:
            progress.close()

    def to_supervisely(
        self,
        item: MeshConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Dict:
        ann_json = deepcopy(item.ann_data) if item.ann_data is not None else None
        if ann_json is None:
            return item.create_empty_annotation().to_json()

        renamed_classes = renamed_classes or {}
        for label in ann_json.get("labels", []):
            class_title = label.get("classTitle")
            if class_title is not None:
                label["classTitle"] = renamed_classes.get(class_title, class_title)
        return ann_json

    def _strip_label_data_from_meshes(self) -> None:
        """Remove baked-in label data from PLY files before upload.

        The annotation is already built from these values in :meth:`validate_format`.
        ``class_id``/``object_id`` vertex properties are pure annotation metadata and
        are always removed. Label paint is removed only from vertices that produced
        annotation labels; original (non-label) vertex colors are preserved. When
        every vertex carries label paint, the color properties are dropped entirely.
        """
        try:
            color_to_class = self._build_color_to_class(self._meta)
        except Exception:
            color_to_class = {}
        for item in self._items:
            try:
                _clean_label_data_from_ply(item.path, color_to_class)
            except Exception as e:
                logger.warning(
                    f"Failed to strip label data from mesh {item.name!r}: {repr(e)}. "
                    "The mesh will be uploaded with baked-in label colors."
                )

    @staticmethod
    def _find_project_dir(input_data: str) -> Optional[str]:
        if os.path.isfile(os.path.join(input_data, "meta.json")):
            return input_data

        candidates = []
        for root, _, files in os.walk(input_data):
            if "meta.json" in files:
                candidates.append(root)
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _has_sly_annotation_sidecars(project_dir: str) -> bool:
        for root, dirs, _ in os.walk(project_dir):
            rel_parts = Path(os.path.relpath(root, project_dir)).parts
            if ANN_DIR_NAME in rel_parts or GEOMETRIES_DIR_NAME in rel_parts:
                return True
            if ANN_DIR_NAME in dirs or GEOMETRIES_DIR_NAME in dirs:
                return True
        return False

    @staticmethod
    def _find_mesh_paths(project_dir: str) -> List[str]:
        mesh_paths = []
        for root, _, files in os.walk(project_dir):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if file_name == "meta.json":
                    continue
                if get_file_ext(full_path).lower() == PLY_EXT:
                    mesh_paths.append(full_path)
        return sorted(mesh_paths)

    @staticmethod
    def _build_color_to_class(meta: ProjectMeta) -> Dict[Tuple[int, int, int], str]:
        color_to_class = {}
        for obj_class in meta.obj_classes:
            if obj_class.geometry_type not in (Mesh, AnyGeometry):
                continue
            color = tuple(_normalize_color(obj_class.color))
            if color in color_to_class:
                raise ValueError(
                    f"Duplicate class color {list(color)} for classes "
                    f"{color_to_class[color]!r} and {obj_class.name!r}"
                )
            color_to_class[color] = obj_class.name
        if len(color_to_class) == 0:
            raise ValueError("Project meta has no mesh-compatible classes with colors.")
        return color_to_class

    @staticmethod
    def _labels_to_annotation(labels: PLYLabels) -> Dict:
        labels_json = []

        for (class_name, object_id), indices in sorted(
            labels.groups.items(), key=lambda item: (item[0][0], item[0][1] or -1)
        ):
            if len(indices) == 0:
                continue
            label_json = {
                "key": uuid.uuid4().hex,
                "classTitle": class_name,
                "tags": [],
                "geometryType": Mesh.geometry_name(),
                "geometry": {"indices": indices},
            }
            if object_id is not None:
                label_json["customData"] = {"sourceObjectId": object_id}
            labels_json.append(label_json)

        return {
            "description": "",
            "key": uuid.uuid4().hex,
            "tags": [],
            "labels": labels_json,
        }


def _clean_label_data_from_ply(mesh_path: str, color_to_class: Dict) -> None:
    """Rewrite an ASCII PLY file in place, removing baked-in label data.

    ``class_id``/``object_id`` vertex properties are always dropped. Vertices whose
    color matches a class color (the same rule the import uses to build labels) get
    their color reset to neutral white; other vertices keep their original colors.
    If every vertex carries label paint, the color (and alpha) properties are
    dropped entirely instead. Faces and other elements are left untouched.
    """
    with open(mesh_path, "r", encoding="ascii") as file:
        lines = file.readlines()

    header_end = None
    current_element = None
    vertex_count = 0
    vertex_properties = []
    property_line_indexes = {}  # vertex property index -> line index in `lines`

    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "end_header":
            header_end = line_index
            break
        parts = stripped.split()
        if len(parts) == 3 and parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif len(parts) == 3 and parts[0] == "property" and current_element == "vertex":
            property_line_indexes[len(vertex_properties)] = line_index
            vertex_properties.append(parts[2])

    if header_end is None:
        raise ValueError("PLY header is missing end_header.")

    drop_columns = set()
    for name in ("class_id", "object_id"):
        index = _property_index(vertex_properties, name)
        if index is not None:
            drop_columns.add(index)

    body_start = header_end + 1
    vertex_lines = lines[body_start : body_start + vertex_count]
    rest_lines = lines[body_start + vertex_count :]

    color_indexes = _get_color_property_indexes(vertex_properties)
    marker_index = _property_index(vertex_properties, "class_id")
    labeled_rows = set()
    if color_indexes is not None:
        for row_index, row in enumerate(vertex_lines):
            values = row.split()
            color = tuple(_parse_color_channel(values[index]) for index in color_indexes)
            if color not in color_to_class:
                continue
            if marker_index is not None and int(float(values[marker_index])) == UNLABELED_ID:
                continue
            labeled_rows.add(row_index)

        if vertex_count > 0 and len(labeled_rows) == vertex_count:
            # Every vertex carries label paint — no original colors survive,
            # drop the color properties entirely.
            drop_columns.update(color_indexes)
            for name in ("alpha", "diffuse_alpha"):
                index = _property_index(vertex_properties, name)
                if index is not None:
                    drop_columns.add(index)
            labeled_rows = set()

    if len(drop_columns) == 0 and len(labeled_rows) == 0:
        return

    new_vertex_lines = []
    for row_index, row in enumerate(vertex_lines):
        values = row.split()
        if row_index in labeled_rows:
            for index in color_indexes:
                values[index] = _NEUTRAL_COLOR
        kept = [value for column, value in enumerate(values) if column not in drop_columns]
        new_vertex_lines.append(" ".join(kept) + "\n")

    dropped_line_indexes = {
        property_line_indexes[column]
        for column in drop_columns
        if column in property_line_indexes
    }
    header_lines = [
        line
        for line_index, line in enumerate(lines[: header_end + 1])
        if line_index not in dropped_line_indexes
    ]

    with open(mesh_path, "w", encoding="ascii") as file:
        file.writelines(header_lines)
        file.writelines(new_vertex_lines)
        file.writelines(rest_lines)


def _read_ascii_ply_labels(
    mesh_path: str,
    color_to_class: Dict[Tuple[int, int, int], str],
) -> PLYLabels:
    with open(mesh_path, "r", encoding="ascii") as file:
        header = _read_ply_header(file)
        if header["format"] != "ascii":
            raise ValueError(f"PLY file {mesh_path!r} is not ASCII.")

        vertex_count = header["vertex_count"]
        vertex_properties = header["vertex_properties"]
        color_indexes = _get_color_property_indexes(vertex_properties)
        if color_indexes is None:
            return PLYLabels(groups={}, has_rgb=False, labeled_vertices=0)

        # class_id and object_id are mandatory parts of the per-vertex format:
        # class_id marks annotated vertices, object_id carries instance grouping.
        class_id_index = _property_index(vertex_properties, "class_id")
        object_id_index = _property_index(vertex_properties, "object_id")
        if class_id_index is None or object_id_index is None:
            return PLYLabels(groups={}, has_rgb=False, labeled_vertices=0)

        groups = {}
        labeled_vertices = 0

        for vertex_index in range(vertex_count):
            line = file.readline()
            if line == "":
                raise ValueError(f"PLY file {mesh_path!r} ended before all vertices were read.")
            values = line.split()
            if len(values) < len(vertex_properties):
                raise ValueError(f"PLY vertex row has fewer values than declared properties.")

            color = tuple(_parse_color_channel(values[index]) for index in color_indexes)
            class_name = color_to_class.get(color)
            if class_name is None:
                continue

            # class_id == -1 explicitly marks the vertex as not annotated. This lets
            # background vertices coexist with a class of the same color (e.g. white).
            if int(float(values[class_id_index])) == UNLABELED_ID:
                continue

            object_id = None
            parsed_object_id = int(float(values[object_id_index]))
            if parsed_object_id != UNLABELED_ID:
                object_id = parsed_object_id

            groups.setdefault((class_name, object_id), []).append(vertex_index)
            labeled_vertices += 1

    return PLYLabels(groups=groups, has_rgb=True, labeled_vertices=labeled_vertices)


def _read_ply_header(file) -> Dict:
    first_line = file.readline().strip()
    if first_line != "ply":
        raise ValueError("File is not a PLY mesh.")

    ply_format = None
    current_element = None
    vertex_count = None
    vertex_properties = []

    while True:
        line = file.readline()
        if line == "":
            raise ValueError("PLY header is missing end_header.")
        line = line.strip()
        if line == "end_header":
            break
        if line == "" or line.startswith("comment "):
            continue

        parts = line.split()
        if parts[0] == "format":
            if len(parts) < 3:
                raise ValueError("Invalid PLY format line.")
            ply_format = parts[1]
            continue

        if parts[0] == "element":
            if len(parts) != 3:
                raise ValueError("Invalid PLY element line.")
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            continue

        if parts[0] == "property" and current_element == "vertex":
            if len(parts) == 3:
                vertex_properties.append(parts[2])
            elif len(parts) >= 5 and parts[1] == "list":
                raise ValueError("PLY vertex list properties are not supported.")

    if vertex_count is None:
        raise ValueError("PLY vertex element is missing.")
    return {
        "format": ply_format,
        "vertex_count": vertex_count,
        "vertex_properties": vertex_properties,
    }


def _get_color_property_indexes(properties: List[str]) -> Optional[Tuple[int, int, int]]:
    for names in (("red", "green", "blue"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        indexes = [_property_index(properties, name) for name in names]
        if all(index is not None for index in indexes):
            return tuple(indexes)
    return None


def _property_index(properties: List[str], name: str) -> Optional[int]:
    try:
        return properties.index(name)
    except ValueError:
        return None


def _normalize_color(color) -> List[int]:
    rgb = [int(color[0]), int(color[1]), int(color[2])]
    for channel in rgb:
        if channel < 0 or channel > 255:
            raise ValueError(f"Color channel {channel!r} is outside 0..255.")
    return rgb


def _parse_color_channel(value: str) -> int:
    return max(0, min(255, int(round(float(value)))))
