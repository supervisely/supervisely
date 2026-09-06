import os
from typing import List, Optional

from supervisely import Annotation, ProjectMeta, logger
from supervisely._utils import batched, is_development
from supervisely.api.api import Api, ApiContext
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.imaging.image import is_valid_ext
from supervisely.io.fs import dirs_filter, get_file_ext, list_files
from supervisely.io.json import load_json_file
from supervisely.project.project_settings import LabelingInterface


class OverlayImageConverter(ImageConverter):
    OVERLAY_DIR = "overlay"
    ITEM_DIR = "img"
    ANN_DIR = "ann"

    class Item:

        def __init__(
            self,
            path: str,
            overlay_paths: List[str],
            ann_path: Optional[str],
        ):
            self._path = path
            self._ann_path = ann_path
            self._overlay_paths = overlay_paths

        @property
        def path(self) -> str:
            return self._path

        @property
        def ann_path(self) -> Optional[str]:
            return self._ann_path

        @property
        def overlay_paths(self) -> List[str]:
            return self._overlay_paths

        def has_annotation(self) -> bool:
            return self._ann_path is not None

    def __str__(self):
        return AvailableImageConverters.OVERLAY

    def validate_labeling_interface(self) -> bool:
        return self._labeling_interface == LabelingInterface.OVERLAY

    def validate_format(self) -> bool:
        def _check_fn(path: str) -> bool:
            if not os.path.isdir(os.path.join(path, self.OVERLAY_DIR)):
                return False
            if not os.path.isdir(os.path.join(path, self.ITEM_DIR)):
                return False
            return True

        items = []
        for dir in dirs_filter(self._input_data, _check_fn):
            overlay_dir = os.path.join(dir, self.OVERLAY_DIR)
            item_dir = os.path.join(dir, self.ITEM_DIR)
            ann_dir = os.path.join(dir, self.ANN_DIR)
            for item in list_files(item_dir):
                item_name, image_ext = os.path.splitext(os.path.basename(item))
                if not is_valid_ext(image_ext):
                    logger.warning("Skipping file with unsupported extension: %s", item)
                    continue

                item_overlay_dir = os.path.join(overlay_dir, item_name)
                if not os.path.isdir(item_overlay_dir):
                    logger.warning("Overlay directory not found for item: %s", item)
                    continue

                img_validation_fn = lambda p: is_valid_ext(get_file_ext(p))
                overlay_paths = [
                    p for p in list_files(item_overlay_dir, filter_fn=img_validation_fn)
                ]
                if not overlay_paths:
                    logger.warning("No valid overlay images found for item: %s", item)
                    continue

                ann_path = os.path.join(ann_dir, os.path.basename(item) + ".json")
                if not os.path.exists(ann_path):
                    logger.warning("Annotation file not found for item: %s", item)
                    ann_path = None

                items.append(self.Item(item, overlay_paths, ann_path))
        self._items = items
        if items:
            logger.info("Found %d items in overlay images format", len(items))

        return bool(items)

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        dataset = api.dataset.get_info_by_id(dataset_id)
        project_id = dataset.project_id
        api.project.set_overlay_settings(project_id)

        meta_json = api.project.get_meta(project_id, with_settings=True)
        meta = ProjectMeta.from_json(meta_json)

        if log_progress:
            progress, progress_cb = self.get_progress(len(self._items), "Uploading images...")
        else:
            progress_cb = None

        with ApiContext(api=api, project_id=project_id, dataset_id=dataset_id, project_meta=meta):
            for items_batch in batched(self._items, batch_size):
                paths = [item.path for item in items_batch]
                overlay_paths = [item.overlay_paths for item in items_batch]
                names = [os.path.basename(item.path) for item in items_batch]
                overlay_names = [
                    [os.path.basename(p) for p in item.overlay_paths] for item in items_batch
                ]

                parent_image_infos, _ = api.image.upload_overlay_images(
                    dataset_id,
                    names=names,
                    paths=paths,
                    overlay_names=overlay_names,
                    overlay_paths=overlay_paths,
                )
                p_name_to_info = {
                    name: info for info, name in zip(parent_image_infos, names)
                }

                id_to_ann_path = {}
                for item in items_batch:
                    if item.has_annotation():
                        info = p_name_to_info[os.path.basename(item.path)]
                        try:
                            id_to_ann_path[info.id] = Annotation.from_json(
                                load_json_file(item.ann_path), meta
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to load annotation for item: %s. Skipping annotation upload. Error: %s",
                                item.path,
                                e,
                            )
                            continue

                if id_to_ann_path:
                    api.annotation.upload_anns(
                        list(id_to_ann_path.keys()), list(id_to_ann_path.values())
                    )

                if log_progress:
                    progress_cb(len(items_batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset '{dataset.name}' has been successfully uploaded.")
