import os
from pathlib import Path

import supervisely.convert.image.sly.sly_image_helper as helper
from supervisely import (
    Annotation,
    Api,
    ImageInfo,
    Label,
    Project,
    ProjectMeta,
    Rectangle,
    batched,
    is_development,
    logger,
)
from supervisely.convert.image.image_converter import ImageConverter
from supervisely.convert.image.image_helper import validate_image_bounds
from supervisely.convert.image.sly.sly_image_converter import SLYImageConverter
from supervisely.io.fs import dir_empty, dir_exists, get_file_ext
from supervisely.io.json import load_json_file


class FastSlyImageConverter(SLYImageConverter, ImageConverter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_links = False

    def validate_format(self) -> bool:

        detected_ann_cnt = 0
        self._items = []
        meta = ProjectMeta()

        for root, _, files in os.walk(self._input_data):
            if Path(root).name == Project.blob_dir_name:
                logger.debug("FastSlyImageConverter: Detected blob directory. Skipping...")
                return False
            for file in files:
                full_path = os.path.join(root, file)
                ext = get_file_ext(full_path)
                if ext == self.ann_ext:
                    ann_json = load_json_file(full_path)
                    if helper.annotation_high_level_validator(ann_json):
                        meta = helper.get_meta_from_annotation(ann_json, meta)
                        h, w = helper.get_image_size_from_annotation(ann_json)
                        image_name = os.path.splitext(os.path.basename(full_path))[0]
                        item = self.Item(image_name)
                        item.ann_data = full_path
                        item.set_shape((h, w))
                        self._items.append(item)
                        detected_ann_cnt += 1
                    elif helper.is_image_info(ann_json):
                        image_name = ann_json.get("name")
                        for item in self._items:
                            if item.name == image_name:
                                break
                        if item.shape != (ann_json.get("height"), ann_json.get("width")):
                            logger.warning(
                                f"Mismatch between image shape in annotation and image info JSON. Skipping image info.",
                                extra={"image_name": image_name},
                            )
                            continue
                        item.custom_data["image_info"] = ann_json

                elif self.is_image(full_path) or ext in helper.nondefault_image_exts:
                    self._items = []
                    return False

        if detected_ann_cnt == 0:
            self._items = []
            return False
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
        if meta is None:
            meta = self._meta

        try:
            ann_json = load_json_file(item.ann_data)
            if "annotation" in ann_json:
                ann_json = ann_json["annotation"]
            if renamed_classes or renamed_tags:
                ann_json = helper.rename_in_json(ann_json, renamed_classes, renamed_tags)
            img_size = ann_json["size"]  # dict { "width": 1280, "height": 720 }
            if "width" not in img_size or "height" not in img_size:
                raise ValueError("Invalid image size in annotation JSON")
            img_size = (img_size["height"], img_size["width"])
            labels = validate_image_bounds(
                [Label.from_json(obj, meta) for obj in ann_json["objects"]],
                Rectangle.from_size(img_size),
            )
            return Annotation.from_json(ann_json, meta).clone(labels=labels)
        except Exception as e:
            logger.warning(f"Failed to convert annotation: {repr(e)}")
            return None

    def upload_dataset(
        self,
        api: Api,
        dataset_id: int,
        batch_size: int = 50,
        log_progress=True,
    ) -> None:
        """Convert Supervisely annootations to Supervisely and append to dataset images."""

        meta, renamed_classes, renamed_tags = self.merge_metas_with_conflicts(api, dataset_id)

        existing_images = {img_info.name: img_info for img_info in api.image.get_list(dataset_id)}

        # * Upload non-existing images with hashes from image info json
        image_infos = {}
        for item in self._items:
            image_info = item.custom_data.get("image_info")
            if image_info is None:
                continue
            image_info = ImageInfo(**image_info)
            if image_info.name in existing_images:
                if (
                    image_info.id == existing_images[image_info.name].id
                    and image_info.hash == existing_images[image_info.name].hash
                ):
                    continue
                else:
                    logger.warning(
                        f"Image '{image_info.name}' already exists in the dataset but has different id or hash. "
                        f"Annotation cannot be uploaded for this image. Please, check the dataset and annotation files."
                    )
                    continue
            image_infos[item.name] = image_info

        if image_infos:
            hashes = [image_info.hash for image_info in image_infos.values()]
            existing_hashes = api.image.check_existing_hashes(hashes=hashes)
            missing_hashes = set(hashes) - set(existing_hashes)
            missing_hash_cnt = len(missing_hashes)
            if not existing_hashes:
                logger.warning(
                    "None of the images from image info JSON exist on the server. "
                    "Cannot upload images from image info JSON."
                )
            elif existing_hashes and missing_hash_cnt > 0:
                logger.warning(
                    "Some images from image info JSON already exist on the server, but some are missing. "
                    "Only images with existing hashes will be uploaded.",
                    extra={
                        "existing hash count": len(existing_hashes),
                        "missing hash count": missing_hash_cnt,
                    },
                )
            image_infos = {
                name: info for name, info in image_infos.items() if info.hash in existing_hashes
            }

            if log_progress:
                progress, progress_cb = self.get_progress(
                    len(image_infos), "Uploading images from image infos..."
                )
            else:
                progress_cb = None

            uploaded_infos = api.image.upload_hashes(
                dataset_id=dataset_id,
                names=list(image_infos.keys()),
                hashes=[image_info.hash for image_info in image_infos.values()],
                metas=[image_info.meta for image_info in image_infos.values()],
                progress_cb=progress_cb,
            )
            uploaded_name_to_info = {info.name: info for info in uploaded_infos}
            existing_images.update(uploaded_name_to_info)

        if len(existing_images) == 0:
            raise RuntimeError(
                "Not found images in the dataset. "
                "Please start the import process from a dataset that contains images, "
                "or upload both images and annotations at once."
            )
        if log_progress:
            progress, progress_cb = self.get_progress(
                self.items_count, "Adding Supervisely annotations..."
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
                    logger.warning(
                        f"Image '{item.name}' has different shapes in JSON file and server."
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
