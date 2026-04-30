import os
from typing import Dict

from supervisely import ProjectMeta, logger
from supervisely.convert.base_converter import AvailableMeshConverters
from supervisely.convert.mesh.mesh_converter import MeshConverter
from supervisely.convert.mesh.sly import sly_mesh_helper
from supervisely.io.json import load_json_file
from supervisely.project.mesh_project import MeshDataset, MeshProject


class SLYMeshConverter(MeshConverter):
    """Imports Supervisely mesh project format from local folder."""

    def __str__(self) -> str:
        return AvailableMeshConverters.SLY

    @property
    def ann_ext(self) -> str:
        return ".json"

    @property
    def key_file_ext(self) -> str:
        return ".json"

    def validate_ann_file(self, ann_path: str, meta: ProjectMeta = None) -> bool:
        try:
            ann_json = self._load_annotation_json(ann_path)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            for key in sly_mesh_helper.SLY_MESH_ANN_KEYS:
                if not isinstance(ann_json.get(key), list):
                    return False
            return True
        except Exception as e:
            logger.warning(f"Failed to validate mesh annotation: {repr(e)}")
            return False

    def validate_key_file(self, key_file_path: str) -> bool:
        try:
            self._meta = ProjectMeta.from_json(load_json_file(key_file_path))
            return True
        except Exception:
            return False

    def validate_format(self) -> bool:
        if self.upload_as_links and self.supports_links:
            self._download_remote_ann_files(exts_to_download=[self.ann_ext, ".bin"])
        return self.read_sly_project(self._input_data)

    def read_sly_project(self, input_data: str) -> bool:
        try:
            project_fs = MeshProject.read_single(input_data)
            self._meta = project_fs.meta
            self._items = []

            for dataset_fs in project_fs.datasets:
                dataset_fs: MeshDataset
                for item_name in dataset_fs:
                    item_paths = dataset_fs.get_item_paths(item_name)
                    ann_path = item_paths.ann_path
                    if not self.validate_ann_file(ann_path, self._meta):
                        logger.warning(
                            f"Mesh annotation for item {item_name!r} is invalid. "
                            "The item will be uploaded with an empty annotation."
                        )
                        ann_path = None
                    item = self.Item(
                        item_path=item_paths.mesh_path,
                        ann_data=ann_path,
                        ann_dir=item_paths.ann_dir,
                        geometries_dir=item_paths.geometries_dir,
                    )
                    self._items.append(item)
            return len(self._items) > 0
        except Exception as e:
            logger.info(f"Failed to read Supervisely mesh project: {repr(e)}")
            return False

    def to_supervisely(
        self,
        item: MeshConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Dict:
        if item.ann_data is None:
            return item.create_empty_annotation().to_json()

        try:
            ann_json = self._load_annotation_json(item.ann_data)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            if renamed_classes or renamed_tags:
                ann_json = sly_mesh_helper.rename_in_json(
                    ann_json, renamed_classes, renamed_tags
                )
            return ann_json
        except Exception as e:
            logger.warning(f"Failed to read mesh annotation: {repr(e)}")
            return item.create_empty_annotation().to_json()

    @staticmethod
    def _load_annotation_json(ann_path: str) -> Dict:
        return MeshDataset._decode_geometry_sidecars_from_dir(
            os.path.dirname(ann_path), load_json_file(ann_path)
        )
