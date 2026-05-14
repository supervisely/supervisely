import os
from typing import List, Optional, Set, Tuple, Union

from supervisely import Api, MeshAnnotation, batched, generate_free_name, is_development, logger
from supervisely.api.mesh.mesh_api import ALLOWED_MESH_EXTENSIONS
from supervisely.convert.base_converter import BaseConverter
from supervisely.project.project_meta import ProjectMeta
from supervisely.video_annotation.key_id_map import KeyIdMap


class MeshConverter(BaseConverter):
    """Base converter for mesh projects."""

    allowed_exts = sorted(ALLOWED_MESH_EXTENSIONS)
    modality = "meshes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = True

    class Item(BaseConverter.BaseItem):
        """Base mesh item: mesh path plus optional raw annotation JSON/path."""

        def __init__(
            self,
            item_path: str,
            ann_data: Union[str, dict] = None,
            shape: Union[Tuple, List] = None,
            custom_data: Optional[dict] = None,
            ann_dir: Optional[str] = None,
            geometries_dir: Optional[str] = None,
        ):
            super().__init__(
                item_path=item_path,
                ann_data=ann_data,
                shape=shape,
                custom_data=custom_data,
            )
            self._type = "mesh"
            self._ann_dir = ann_dir
            self._geometries_dir = geometries_dir

        @property
        def ann_dir(self) -> Optional[str]:
            return self._ann_dir

        @property
        def geometries_dir(self) -> Optional[str]:
            return self._geometries_dir

        def create_empty_annotation(self) -> MeshAnnotation:
            return MeshAnnotation()

    @property
    def format(self):
        return self._converter.format if self._converter is not None else self.__str__()

    def __str__(self) -> str:
        return self.modality

    @property
    def ann_ext(self):
        return None

    @property
    def key_file_ext(self):
        return None

    @staticmethod
    def validate_ann_file(ann_path, meta=None):
        return False

    @staticmethod
    def validate_key_file(key_file_path=None):
        return False

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 10,
        log_progress=True,
    ) -> None:
        """Upload converted mesh data and mesh annotation JSONs to Supervisely."""

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_mesh_infos = api.mesh.get_list(dataset_id)
        existing_mesh_names = set(mesh.name for mesh in existing_mesh_infos)

        if log_progress:
            progress, progress_cb = self.get_progress(self.items_count, "Uploading meshes...")
        else:
            progress_cb = None

        upload_fn = api.mesh.upload_paths
        if self.upload_as_links and self.supports_links:
            upload_fn = api.mesh.upload_links

        key_id_map = KeyIdMap()
        for batch in batched(self._items, batch_size=batch_size):
            item_names = []
            item_paths = []
            anns = []

            for item in batch:
                item.name = generate_free_name(
                    existing_mesh_names, item.name, with_ext=True, extend_used_names=True
                )
                item_names.append(item.name)
                if self.upload_as_links:
                    item_paths.append(
                        self.remote_files_map.get(os.path.abspath(item.path), item.path)
                    )
                else:
                    item_paths.append(item.path)

                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                anns.append(ann)

            mesh_infos = upload_fn(dataset_id, item_names, item_paths)
            for mesh_info, ann in zip(mesh_infos, anns):
                api.mesh.annotation.append(mesh_info.id, ann, key_id_map=key_id_map)

            if log_progress:
                progress_cb(len(batch))

        if log_progress and is_development():
            progress.close()
        logger.info(f"Dataset ID:{dataset_id} has been successfully uploaded.")

    def _collect_items_if_format_not_detected(self) -> Tuple[List[Item], bool, Set[str]]:
        only_modality_items = True
        unsupported_exts = set()
        items = []

        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                if file in {"meta.json", "key_id_map.json"}:
                    continue
                ext = os.path.splitext(full_path)[1].lower()
                if ext in self.allowed_exts:
                    items.append(self.Item(full_path))
                else:
                    only_modality_items = False
                    if ext in self.unsupported_exts:
                        unsupported_exts.add(ext)

        return items, only_modality_items, unsupported_exts
