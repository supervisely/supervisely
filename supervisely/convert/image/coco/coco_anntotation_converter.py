import os

import supervisely.convert.image.coco.coco_helper as coco_helper
from supervisely import Annotation, Api, ProjectMeta, batched, is_development, logger
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.coco.coco_converter import COCOConverter
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.io.fs import get_file_ext

COCO_ANN_KEYS = ["images", "annotations"]


class FastCOCOConverter(COCOConverter, ImageConverter):

    def __str__(self) -> str:
        return AvailableImageConverters.FAST_COCO

    def validate_format(self) -> bool:
        from pycocotools.coco import COCO  # pylint: disable=import-error

        detected_ann_cnt = 0
        ann_paths = []
        for root, _, files in os.walk(self._input_data):
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == self.ann_ext:
                    ann_paths.append(full_path)
                elif self.is_image(full_path):
                    return False

        if len(ann_paths) == 0:
            return False

        # create Items
        self._items = []
        meta = ProjectMeta()
        for ann_path in ann_paths:
            try:
                with coco_helper.HiddenCocoPrints():
                    coco = COCO(ann_path)
            except:
                continue
            if not all(key in coco.dataset for key in COCO_ANN_KEYS):
                continue
            coco_anns = coco.imgToAnns
            coco_images = coco.imgs
            if len(coco.cats) > 0:
                coco_categories = coco.loadCats(ids=coco.getCatIds())
            else:
                coco_categories = []
            self._coco_categories.extend(coco_categories)
            coco_items = coco_images.items()
            meta = self.generate_meta_from_annotation(coco, meta)
            # create ann dict
            for image_id, image_info in coco_items:
                image_name = image_info.get("file_name", image_info.get("name"))
                image_url = image_info.get(
                    "coco_url",
                    image_info.get(
                        "flickr_url", image_info.get("url", image_info.get("path", None))
                    ),
                )
                width = image_info.get("width")
                height = image_info.get("height")

                coco_ann = coco_anns[image_id]
                if len(coco_ann) == 0 or coco_ann is None or image_name is None:
                    continue
                item = self.Item(image_name) if image_url is None else self.Item(image_url)
                item.name = image_name
                item.ann_data = coco_ann
                item.set_shape((height, width))
                self._items.append(item)
                detected_ann_cnt += len(coco_ann)

        self._meta = meta
        return detected_ann_cnt > 0

    def to_supervisely(
        self,
        item: ImageConverter.Item,
        meta: ProjectMeta = None,
        renamed_classes: dict = None,
        renamed_tags: dict = None,
    ) -> Annotation:
        """Convert to Supervisely format."""
        try:
            ann = coco_helper.create_supervisely_annotation(
                item,
                meta,
                self._coco_categories,
                renamed_classes,
                renamed_tags,
            )
            return ann
        except Exception as e:
            logger.error(
                f"Error during conversion of annotation for image '{item.name}': {repr(e)}"
            )
            return None

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Convert COCO annootations to Supervisely and append to dataset images."""

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_images = {img_info.name: img_info for img_info in api.image.get_list(dataset_id)}
        if len(existing_images) == 0:
            raise RuntimeError(
                "Failed to upload COCO annotations, you have no images in the dataset. "
                "To add annotations to your dataset, please start import process from dataset with images. "
                "Or you can upload images and annotations together. "
            )
        if log_progress:
            progress, progress_cb = self.get_progress(
                self.items_count, "Adding COCO annotations..."
            )
        else:
            progress_cb = None

        for batch in batched(self._items, batch_size=batch_size):
            img_ids = []
            anns = []
            for item in batch:
                existing_image = existing_images.get(item.name)
                if existing_image is None:
                    continue
                if item.shape != (existing_image.height, existing_image.width):
                    logger.warn(
                        f"Image '{item.name}' has different shapes in COCO annotation and Supervisely."
                    )
                    continue

                ann = self.to_supervisely(item, meta, renamed_classes, renamed_tags)
                if ann is not None:
                    img_ids.append(existing_image.id)
                    anns.append(ann)

            if len(anns) == len(img_ids):
                existing_anns = api.annotation.download_json_batch(dataset_id, img_ids)
                existing_anns = [Annotation.from_json(json, meta) for json in existing_anns]
                merged_anns = []
                for existing_ann, new_ann in zip(existing_anns, anns):
                    merged_ann = existing_ann.merge(new_ann)
                    merged_anns.append(merged_ann)

                api.annotation.upload_anns(img_ids, merged_anns)

            if log_progress:
                progress_cb(len(batch))

        if log_progress:
            if is_development():
                progress.close()
        logger.info(f"Dataset ID:'{dataset_id}' has been successfully uploaded.")
